import time
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolo import YOLO, center, yolo_detections_to_norfair_detections

import norfair
from norfair import Detection, Paths, Video
from norfair.tracker import TrackedObject

from threading import Thread
from numpy_ringbuffer import RingBuffer

import numpy as np

import face_tracking as ft

# detection parameters
CONF_THRESHOLD: float = 0.6
IOU_THRESHOLD: float = 0.4
IMAGE_SIZE: int = 720

# tracking parameters
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

BUF_LEN = 15
FACE_PERIOD = 1


def analyze_face(obj, frame):
    print('processing ', obj.id)
    x1, y1, x2, y2 = obj.box
    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
        gender, emotion, reg = ft.get_face(frame[int(y1):int(y2), int(x1):int(x2)])
        params = {'emotion': emotion, 'gender': gender}
        obj.set_params(params)


class Object:
    def __init__(self, o: TrackedObject):
        self.id = o.global_id
        det = o.last_detection
        self.box = det.points[0][0], det.points[0][1], det.points[1][0], det.points[1][1]
        self.class_id = det.label
        self.speed = 0
        self.frame_num = 0
        self.params = dict()
        self.params_changed = None
        self.face_thread = None
        self.face_time = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def copy_attr(self, obj):
        self.params = obj.params
        self.params_changed = obj.params_changed
        self.face_thread = obj.face_thread
        self.face_time = obj.face_time

    def set_params(self, p):
        if self.params != p:
            if self.params_changed is not None:
                self.params_changed(self.id, p)
            self.params = p

    @property
    def pos(self):
        return [int(self.box[0] + self.box[2]) / 2, int(self.box[1] + self.box[3]) / 2]

    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    @property
    def can_process_face(self):
        if self.class_id != 0:
            return False
        if time.time() - self.face_time < FACE_PERIOD:
            return False
        if self.face_thread is None:
            return True
        return not self.face_thread.is_alive()

    def get_distance(self, obj):
        return cv2.norm(self.pos[0] - obj.pos[0], self.pos[1] - obj.pos[1])

    def process_face(self, frame):
        self.face_time = time.time()
        self.face_thread = Thread(target=analyze_face, args=(self, frame), daemon=True)
        self.face_thread.start()


class Tracker:
    def __init__(self, source='0'):

        self.tracks = None
        self.model = YOLO('./yolov7-tiny.pt')
        self.names = self.model.model.names

        # TODO: youtube download
        self.video = Video(camera=int(source)) if source.isnumeric() \
            else Video(input_path=source)
        self.video_iter = iter(self.video)
        frame = next(self.video_iter)
        self.video_buffer = RingBuffer(capacity=BUF_LEN, dtype=(frame.dtype, frame.shape))
        self.frame_num = 0

        self.tracker = norfair.Tracker(
            distance_function='iou',
            distance_threshold=DISTANCE_THRESHOLD_BBOX
        )

        self.__prev_objs = set()
        self.all_objs = dict()
        self.new_objs = None
        self.del_objs = None

    @property
    def video_size(self):
        cap = self.video.video_capture
        w, h = 0, 0
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return w, h

    @property
    def video_fps(self):
        cap = self.video.video_capture
        fps = 0
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
        return fps

    @property
    def video_area(self):
        size = self.video_size
        return size[0] * size[1]

    def get_class_index(self, name):
        return self.names.index(name)

    def get_class_name(self, id):
        return self.names[id]

    def read_video(self):
        try:
            frame = next(self.video_iter)
            self.video_buffer.append(frame)
            norfair.draw_tracked_boxes(frame, self.tracks)
            self.video.show(frame)
        except StopIteration:
            pass

    def track(self, frame, classes=None):
        frame_num = self.frame_num
        self.frame_num += 1

        # inference
        yolo_detections = self.model(frame,
                                     conf_threshold=CONF_THRESHOLD,
                                     iou_threshold=IOU_THRESHOLD,
                                     image_size=IMAGE_SIZE,
                                     classes=classes)
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points='bbox')

        curr_objs = set()

        self.tracks = self.tracker.update(detections)

        for track in self.tracks:
            active_obj = Object(track)
            active_obj.frame_num = frame_num
            id = active_obj.id

            # find out if object already exists, calculate velocity and copy params
            if id in self.all_objs:
                ex_obj = self.all_objs[id]
                # velocity = difference in position over one frame
                active_obj.speed = active_obj.get_distance(ex_obj) / (active_obj.frame_num - ex_obj.frame_num)
                active_obj.copy_attr(ex_obj)

            # face tracking
            if active_obj.can_process_face:
                active_obj.process_face(frame)

            self.all_objs[id] = active_obj
            curr_objs.add(active_obj)

        self.new_objs = curr_objs - self.__prev_objs
        self.del_objs = self.__prev_objs - curr_objs
        self.__prev_objs = curr_objs
        for o in self.del_objs:
            self.all_objs.pop(o.id)
        return curr_objs, self.new_objs, self.del_objs
