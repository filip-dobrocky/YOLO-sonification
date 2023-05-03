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
from collections import deque

import numpy as np

import face_tracking as ft


FACE_PERIOD = 5


def analyze_face(obj, frame):
    x1, y1, x2, y2 = obj.box
    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
        gender, emotion, reg = ft.get_face(frame[int(y1):int(y2), int(x1):int(x2)])
        params = {'emotion': emotion, 'gender': gender}
        obj.set_params(params)


def get_tracker():
    return tracker


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
    def x(self):
        return int(self.box[0] + self.box[2]) / 2

    @property
    def y(self):
        return int(self.box[1] + self.box[3]) / 2

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
    running = False
    video = None
    video_iter = None
    video_buffer = None
    process_queue = None
    frame_counter = 0
    last_frame_num = -1

    def __init__(self, source=None,
                 conf_threshold: float = 0.6,
                 iou_threshold: float = 0.5,
                 image_size: int = 640,
                 dist_threshold: float = 0.85,
                 buf_len: int = 3):

        self.buf_len = buf_len
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        self.tracks = None
        self.model = YOLO('./yolov7-e6e.pt')
        self.names = self.model.model.names
        self.classes = None

        if source is not None:
            self.load_video(source)

        self.tracker = norfair.Tracker(
            distance_function='iou',
            distance_threshold=dist_threshold,
            hit_counter_max=6
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
        if self.video is None:
            return 0
        cap = self.video.video_capture
        fps = 0
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
        return fps

    @property
    def video_frames(self):
        if self.video is None:
            return 0
        cap = self.video.video_capture
        frames = 0
        if cap.isOpened():
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frames

    @property
    def video_area(self):
        size = self.video_size
        return size[0] * size[1]

    @property
    def video_progress(self):
        return (self.frame_counter / self.video_frames) * 100

    def get_class_index(self, name):
        return self.names.index(name)

    def get_class_name(self, id):
        return self.names[id]

    def load_video(self, source):
        # TODO: youtube download
        self.video = Video(camera=int(source)) if source.isnumeric() \
            else Video(input_path=source)
        self.video_iter = iter(self.video)
        self.video_buffer = deque()
        self.process_queue = deque(maxlen=self.buf_len)
        self.frame_counter = 0
        self.last_frame_num = -1
        self.running = True

    def set_video_pos(self, pos: float):
        if self.video is None:
            return
        was_running = self.running
        self.running = False
        index = int(pos*self.video_frames)
        cap = self.video.video_capture
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.frame_counter = index
        self.running = True
        time.sleep(0.1)
        self.running = was_running

    def read_video(self):
        if self.video is None or len(self.video_buffer) >= self.buf_len:
            return False
        try:
            frame = next(self.video_iter)
            self.video_buffer.append((frame, self.frame_counter))
            self.process_queue.append((frame, self.frame_counter))
            self.frame_counter += 1
        except StopIteration:
            print('Video ended.')
            self.running = False
            return False
        return True

    def show_video(self):
        if len(self.video_buffer) < self.buf_len:
            return False
        frame = self.video_buffer.popleft()[0]
        norfair.draw_boxes(frame, self.tracks, draw_labels=True, draw_ids=True)
        self.video.show(frame)
        return True

    def track(self):
        if len(self.process_queue):
            tmp = self.process_queue.popleft()
            frame = tmp[0]
            frame_num = tmp[1]
        else:
            return None

        tracker_period = frame_num - self.last_frame_num
        self.last_frame_num = frame_num

        # print(tracker_period)

        # inference
        yolo_detections = self.model(frame,
                                     conf_threshold=self.conf_threshold,
                                     iou_threshold=self.iou_threshold,
                                     image_size=self.image_size,
                                     classes=self.classes)
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points='bbox')

        curr_objs = set()
        self.tracks = self.tracker.update(detections, period=tracker_period)

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
