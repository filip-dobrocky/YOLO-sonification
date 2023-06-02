import time
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolo import YOLO, center, yolo_detections_to_norfair_detections

import norfair
from norfair import Detection, Paths, Video
from norfair.tracker import TrackedObject

from threading import Thread
from collections import deque

import numpy as np

import face_tracking as ft

FACE_PERIOD = 0.02   # seconds
MAX_VID_SIZE = [1280, 720]

EMO_IMAGES = dict()
for e in ft.emotion_labels:
    EMO_IMAGES[e] = cv2.imread('./images/' + e + '.png', cv2.IMREAD_UNCHANGED)

face_threads = 0
MAX_FACE_THREADS = 5


def clip_vid_size(w, h):
    max_w, max_h = MAX_VID_SIZE
    dim = (w, h)
    if h > max_h:
        r = max_h / float(h)
        dim = (int(w * r), max_h)
    elif w > max_w:
        r = max_w / float(w)
        dim = (max_w, int(h * r))
    return dim


def clip_vid_frame(frame):
    (h, w) = frame.shape[:2]
    dim = clip_vid_size(w, h)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def analyze_face(obj, frame):
    global face_threads
    face_threads += 1
    x1, y1, x2, y2 = obj.box
    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
        result = ft.get_face(frame[int(y1):int(y2), int(x1):int(x2)].copy())
        if result is not None:
            gender, emotion, reg, conf = result
            if conf > 60:
                obj.params = {'emotion': emotion, 'gender': gender, 'face_reg': reg}
    face_threads -= 1


class Object:
    def __init__(self, o: TrackedObject):
        self.id = o.global_id
        det = o.last_detection
        self.box = det.points[0][0], det.points[0][1], det.points[1][0], det.points[1][1]
        self.class_id = det.label
        self.speed = 0
        self.frame_num = 0
        self.params = dict()
        self.old_params = dict()
        self.face_thread = None
        self.face_time = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def copy_attr(self, obj):
        self.params = obj.params
        self.old_params = obj.old_params
        self.face_thread = obj.face_thread
        self.face_time = obj.face_time

    def params_changed(self):
        if 'emotion' not in self.old_params:
            changed = 'emotion' in self.params
        else:
            changed = self.old_params['emotion'] != self.params['emotion'] or \
                  self.old_params['gender'] != self.params['gender']
        self.old_params = self.params
        return changed

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
    def sex_id(self):
        if 'gender' in self.params:
            return ft.gender_labels.index(self.params['gender'])
        return 0

    @property
    def emo_id(self):
        if 'emotion' in self.params:
            return ft.emotion_labels.index(self.params['emotion'])
        return 0

    @property
    def can_process_face(self):
        if self.class_id != 0:
            return False
        if face_threads >= MAX_FACE_THREADS:
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
    display_boxes = True
    display_emotions = True
    detect_faces = False
    moving_cam = False

    def __init__(self, source=None,
                 conf_threshold: float = 0.6,
                 iou_threshold: float = 0.3,
                 image_size: int = 640,
                 dist_threshold: float = 0.85,
                 buf_len: int = 3):

        self.buf_len = buf_len
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.dist_threshold = dist_threshold

        self.tracks = None
        self.model = YOLO('./yolov7-e6e.pt')
        self.names = self.model.model.names
        self.classes = None
        self.motion_estimator = None
        self.tracker = None

        if source is not None:
            self.load_video(source)

        self.__prev_objs = set()
        self.all_objs = dict()
        self.curr_objs = set()
        self.new_objs = set()
        self.del_objs = set()

    @property
    def video_size(self):
        cap = self.video.video_capture
        w, h = 1, 1
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return clip_vid_size(w, h)

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
        return (self.frame_counter / self.video_frames) * 100 if self.video_frames else 0

    def get_class_index(self, name):
        return self.names.index(name)

    def get_class_name(self, id):
        return self.names[id]

    def reset_tracker(self):
        self.video_buffer = deque()
        self.process_queue = deque(maxlen=self.buf_len)
        self.tracker = norfair.Tracker(
            distance_function='iou',
            distance_threshold=self.dist_threshold,
            hit_counter_max=6
        )
        self.motion_estimator = norfair.camera_motion.MotionEstimator()

    def load_video(self, source):
        self.running = False
        self.reset_tracker()
        self.video = Video(camera=int(source)) if source.isnumeric() \
            else Video(input_path=source)
        self.video_iter = iter(self.video)
        self.frame_counter = 0
        self.last_frame_num = -1
        self.running = True

    def set_video_pos(self, pos: float):
        if self.video is None:
            return
        was_running = self.running
        self.running = False
        index = int(pos * self.video_frames)
        cap = self.video.video_capture
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.frame_counter = index
        self.reset_tracker()
        self.running = True
        time.sleep(0.1)
        self.running = was_running

    def read_video(self):
        if self.video is None or len(self.video_buffer) >= self.buf_len:
            return False
        try:
            frame = next(self.video_iter)
            frame = clip_vid_frame(frame)
            self.video_buffer.append((frame, self.frame_counter))
            self.process_queue.append((frame, self.frame_counter))
            self.frame_counter += 1
        except StopIteration:
            print('Video ended.')
            self.running = False
            return False
        return True

    def draw_emotions(self, frame):
        for o in self.curr_objs.copy():
            if not o.params:
                continue
            rect = o.params['face_reg']
            x, y, w, h = int(rect['x'] + o.box[0]), int(rect['y'] + o.box[1]), int(rect['w']), int(rect['h'])
            img = EMO_IMAGES[o.params['emotion']].copy()
            img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
            alpha_s = img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 200, 200), 1, cv2.LINE_AA)
            try:
                for c in range(3):
                    frame[y:y + h, x:x + w, c] = (alpha_s * (200 - img[:, :, c]) + alpha_l * frame[y:y + h, x:x + w, c])
            except ValueError:
                pass

    def show_video(self):
        if len(self.video_buffer) < self.buf_len:
            return False
        frame = self.video_buffer.popleft()[0].copy()
        if self.display_boxes:
            norfair.draw_boxes(frame, self.tracks, draw_labels=True, draw_ids=False)
        if self.display_emotions:
            self.draw_emotions(frame)
        self.video.show(frame)
        return True

    def get_cam_transformation(self, frame):
        trans = None
        if self.moving_cam and self.motion_estimator is not None:
            try:
                trans = self.motion_estimator.update(frame)
            except:
                self.reset_tracker()
        return trans

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

        self.curr_objs = set()
        self.tracks = self.tracker.update(detections, period=tracker_period,
                                          coord_transformations=self.get_cam_transformation(frame))

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
                self.all_objs[id].__dict__ = active_obj.__dict__.copy()
            else:
                self.all_objs[id] = active_obj

            # face tracking
            if self.detect_faces:
                if self.all_objs[id].can_process_face:
                    self.all_objs[id].process_face(frame)
            elif len(self.all_objs[id].params):
                self.all_objs[id].params = {}

            self.curr_objs.add(self.all_objs[id])

        self.new_objs = self.curr_objs - self.__prev_objs
        self.del_objs = self.__prev_objs - self.curr_objs
        self.__prev_objs = self.curr_objs
        for o in self.del_objs:
            self.all_objs.pop(o.id)
        return self.curr_objs, self.new_objs, self.del_objs
