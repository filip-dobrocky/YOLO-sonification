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
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync#, load_classifier

import numpy as np
from motpy import Detection, MultiObjectTracker, Track

class Object(Track):
    def __new__(cls, t: Track):
        self = super(Object, cls).__new__(cls, t.id, t.box, t.score, t.class_id)
        self.speed = 0
        self.frame = 0
        return self

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    @property
    def pos(self):
        return [int(self.box[0] + self.box[2]) / 2, int(self.box[1] + self.box[3]) / 2]

    @property
    def size(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    def get_distance(self, obj):
        return cv2.norm(self.pos[0] - obj.pos[0], self.pos[1] - obj.pos[1])

class Tracker:
    def __init__(self, source = '0',
                       weights = 'yolov5n.pt',
                       imgsz = 320,
                       device = 'cpu',
                       augment = True,
                       conf_thres = 0.6,
                       iou_thres = 0.6,
                       agnostic_nms = True):
        self.is_video = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        self.weights, self.augment, self.conf_thres, self.iou_thres, self.agnostic_nms \
            = weights, augment, conf_thres, iou_thres, agnostic_nms

        # Initialize
        set_logging()
        self.__device = select_device(device)
        self.__half = self.__device.type != 'cpu'  # half precision only supported on CUDA

        # Create a multi object tracker
        self.__tracker = MultiObjectTracker(dt=0.1)

        # Load model
        self.model = attempt_load(weights, device='cpu')  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.__half:
            self.model.half()  # to FP16

        # Second-stage classifier
        # self.classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        if self.is_video:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            self.save_img = True
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.__device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz)
                       .to(self.__device).type_as(next(self.model.parameters())))  # run once

        self.__prev_objs = set()
        self.all_objs = dict()
        self.new_objs = None
        self.del_objs = None


    @property
    def video_size(self):
        return self.dataset.get_video_size(0)
    def get_class_index(self, name):
        return list(self.names.keys())[list(self.names.values()).index(name)]

    def get_class_name(self, id):
        return self.names[id]

    def track(self, classes = None):
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

        # Apply Classifier
        # if self.classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        curr_objs = set()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.is_video:  # batch_size >= 1
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
                tracks = self.__tracker.active_tracks(3)

                for track in tracks:
                    new_obj = Object(track)
                    new_obj.frame = frame

                    try:
                        ex_obj = self.all_objs[new_obj.id]
                        new_obj.speed = new_obj.get_distance(ex_obj) / (new_obj.frame - ex_obj.frame)
                        self.all_objs[new_obj.id] = new_obj
                    except KeyError:
                        self.all_objs[new_obj.id] = new_obj
                    curr_objs.add(new_obj)

                    label = f'{track.id[:5]}: {self.get_class_name(track.class_id)}'
                    annotator = Annotator(im0, line_width=2, pil=not ascii)
                    annotator.box_label(track.box, label, color=colors(track.class_id, True))
                    im0 = annotator.result()

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if self.view_img:
                cv2.imshow('result', im0)
                cv2.waitKey(1)  # 1 millisecond

        self.new_objs = curr_objs - self.__prev_objs
        self.del_objs = self.__prev_objs - curr_objs
        self.__prev_objs = curr_objs
        return 0


if __name__ == '__main__':
    check_requirements()
    #with torch.no_grad():

    #    strip_optimizer(tracker.weights)

    t0 = time.time()
    #tracker = Tracker(source='https://www.youtube.com/watch?v=EUUT1CW_9cg')
    #class_index = tracker.get_class_index('car')
    #print("class: " + str(class_index))
    tracker = Tracker(source='0')
    print(tracker.video_size)
    while True:
        try:
            tracker.track()
            #print("New objects: " + str(tracker.new_objs))
            #print("Expired objects: " + str(tracker.del_objs))
            #for o in tracker.all_objs.values():
            #    print("Speed " + str(o.speed))
        except StopIteration:
            print(f'Done. ({time.time() - t0:.3f}s)')
            break


