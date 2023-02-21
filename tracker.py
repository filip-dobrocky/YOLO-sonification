import sys

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from yolo import YOLO, center, yolo_detections_to_norfair_detections

import norfair
from norfair import Detection, Paths, Video
from norfair.tracker import TrackedObject

import numpy as np

from face_tracking import FaceTracker

# detection parameters
CONF_THRESHOLD: float = 0.6
IOU_THRESHOLD: float = 0.4
IMAGE_SIZE: int = 640

# tracking parameters
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


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

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

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

    def get_distance(self, obj):
        return cv2.norm(self.pos[0] - obj.pos[0], self.pos[1] - obj.pos[1])


class Tracker:
    def __init__(self, source='0'):

        self.model = YOLO('./yolov7.pt')
        self.names = self.model.model.names

        # TODO: youtube download
        self.video = Video(camera=int(source)) if source.isnumeric() \
            else Video(input_path=source)
        self.video_iter = iter(self.video)

        self.tracker = norfair.Tracker(
            distance_function='iou',
            distance_threshold=DISTANCE_THRESHOLD_BBOX
        )

        self.__prev_objs = set()
        self.all_objs = dict()
        self.new_objs = None
        self.del_objs = None

        self.face_tracker = FaceTracker()

    @property
    def video_size(self):
        cap = self.video.video_capture
        w, h = 0, 0
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return w, h

    @property
    def video_area(self):
        size = self.video_size
        return size[0] * size[1]

    def get_class_index(self, name):
        return self.names.index(name)

    def get_class_name(self, id):
        return self.names[id]

    def track(self, classes=None):
        # Run inference
        frame = next(self.video_iter)
        frame_num = self.video.frame_counter
        yolo_detections = self.model(frame,
                                     conf_threshold=CONF_THRESHOLD,
                                     iou_threshold=IOU_THRESHOLD,
                                     image_size=IMAGE_SIZE)
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points='bbox')

        curr_objs = set()

        tracks = self.tracker.update(detections)

        norfair.draw_tracked_boxes(frame, tracks)
        self.video.show(frame)

        for track in tracks:
            active_obj = Object(track)
            active_obj.frame_num = frame_num
            id = active_obj.id

            # find out if object already exists, calculate velocity and copy params
            if id in self.all_objs:
                ex_obj = self.all_objs[id]
                # velocity = difference in position over one frame
                active_obj.speed = active_obj.get_distance(ex_obj) / (active_obj.frame_num - ex_obj.frame_num)
                active_obj.params = ex_obj.params
                active_obj.params_changed = ex_obj.params_changed

            # face tracking
            ft_freq = 10  # track every ft_freq frames
            if active_obj.class_id == self.get_class_index('person'):
                x1, y1, x2, y2 = active_obj.box
                if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0 and active_obj.frame_num % ft_freq == 0:
                    gender, emotion, reg = self.face_tracker.get_face(frame[int(y1):int(y2), int(x1):int(x2)])
                    params = {'emotion': emotion, 'gender': gender}
                    active_obj.set_params(params)

            self.all_objs[id] = active_obj
            curr_objs.add(active_obj)

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
