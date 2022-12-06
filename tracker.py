import sys

sys.path.append('yolov5')

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.dataloaders import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync  # , load_classifier

import numpy as np
from motpy import Detection, MultiObjectTracker, Track

from face_tracking import FaceTracker


class Object(Track):
    def __new__(cls, t: Track):
        self = super(Object, cls).__new__(cls, t.id, t.box, t.score, t.class_id)
        self.speed = 0
        self.frame = 0
        self.params = dict()
        return self

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    @property
    def pos(self):
        return [int(self.box[0] + self.box[2]) / 2, int(self.box[1] + self.box[3]) / 2]

    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    def get_distance(self, obj):
        return cv2.norm(self.pos[0] - obj.pos[0], self.pos[1] - obj.pos[1])


class Tracker:
    def __init__(self, source='0',
                 weights='yolov5n.pt',
                 imgsz=320,
                 augment=True,
                 conf_thres=0.5,
                 iou_thres=0.6,
                 agnostic_nms=True):
        self.is_stream = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        self.weights, self.augment, self.conf_thres, self.iou_thres, self.agnostic_nms \
            = weights, augment, conf_thres, iou_thres, agnostic_nms

        # Initialize
        # set_logging()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__half = self.__device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, device=self.__device.type)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.__half:
            self.model.half()  # to FP16

        # Set Dataloader
        if self.is_stream:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            self.view_img = check_imshow()
            self.save_img = True
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.__device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz)
                       .to(self.__device).type_as(next(self.model.parameters())))  # run once

        # Create a multi object tracker
        self.__tracker = MultiObjectTracker(
            dt=0.1,
            tracker_kwargs={'max_staleness': 2},
            model_spec={'order_pos': 1, 'dim_pos': 2,
                        'order_size': 0, 'dim_size': 2,
                        'q_var_pos': 5000., 'r_var_pos': 0.1},
            matching_fn_kwargs={'min_iou': 0.4,
                                'multi_match_min_iou': 0.93})

        self.__prev_objs = set()
        self.all_objs = dict()
        self.new_objs = None
        self.del_objs = None

        self.face_tracker = FaceTracker()

    @property
    def video_size(self):
        return self.dataset.get_video_size(0)

    @property
    def video_area(self):
        size = self.dataset.get_video_size(0)
        return size[0] * size[1]

    def get_class_index(self, name):
        return list(self.names.keys())[list(self.names.values()).index(name)]

    def get_class_name(self, id):
        return self.names[id]

    def track(self, classes=None):
        # Run inference
        path, img, im0s, vid_cap, *aux = next(self.dataset)
        img = torch.from_numpy(img).to(self.__device)
        img = img.half() if self.__half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=classes,
                                   agnostic=self.agnostic_nms)
        t2 = time_sync()

        curr_objs = set()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.is_stream:  # batch_size >= 1
                s, im0, frame = '%g: ' % i, im0s[i].copy(), self.dataset.count
            else:
                s, im0, frame = '', im0s, getattr(self.dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                out_detections = []
                for *xyxy, conf, cls in reversed(det):
                    object_box = np.array([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    out_detections.append(Detection(box=object_box, score=conf.to('cpu'), class_id=int(cls)))

                self.__tracker.step(out_detections)
                tracks = self.__tracker.active_tracks(min_steps_alive=3)

                for track in tracks:
                    annotator = Annotator(im0, line_width=2, pil=not ascii)

                    active_obj = Object(track)
                    active_obj.frame = frame
                    id = active_obj.id

                    # find out if object already exists, calc velocity and copy params
                    if id in self.all_objs:
                        ex_obj = self.all_objs[id]
                        # velocity = difference in position over one frame
                        active_obj.speed = active_obj.get_distance(ex_obj) / (active_obj.frame - ex_obj.frame)
                        active_obj.params = ex_obj.params

                    # face tracking
                    # ft_freq = 10    # track every ft_freq frames
                    # if active_obj.class_id == self.get_class_index('person'):  # and active_obj.frame % 5 == 0:
                    #     x1, y1, x2, y2 = active_obj.box
                    #     if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0 and active_obj.frame % ft_freq == 0:
                    #         emotion, reg = self.face_tracker.get_emotion(im0[int(y1):int(y2), int(x1):int(x2)])
                    #         active_obj.params['emotion'] = emotion

                    label = f'{track.id[:5]}: {self.get_class_name(track.class_id)} {active_obj.params}'
                    annotator.box_label(track.box, label, color=colors(track.class_id, True))
                    im0 = annotator.result()

                    self.all_objs[id] = active_obj
                    curr_objs.add(active_obj)

                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if self.view_img:
                cv2.imshow('result', im0)
                cv2.waitKey(1)  # 1 millisecond

        self.new_objs = curr_objs - self.__prev_objs
        self.del_objs = self.__prev_objs - curr_objs
        self.__prev_objs = curr_objs
        for o in self.del_objs:
            self.all_objs.pop(o.id)
        return curr_objs, self.new_objs, self.del_objs


if __name__ == '__main__':
    check_requirements()
    # with torch.no_grad():

    #    strip_optimizer(tracker.weights)

    t0 = time.time()
    # tracker = Tracker(source='https://www.youtube.com/watch?v=EUUT1CW_9cg')
    # class_index = tracker.get_class_index('car')
    # print("class: " + str(class_index))
    tracker = Tracker(source='0')
    for i in range(0, 80):
        print(tracker.get_class_name(i))
    print(tracker.video_size)
    while True:
        try:
            tracker.track()
            # print("New objects: " + str(tracker.new_objs))
            # print("Expired objects: " + str(tracker.del_objs))
            # for o in tracker.all_objs.values():
            #    print("Speed " + str(o.speed))
        except StopIteration:
            print(f'Done. ({time.time() - t0:.3f}s)')
            break
