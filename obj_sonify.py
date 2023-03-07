import supriya
import synths
from tracker import Tracker, Object
from mapping import SynthMapping, ParameterMapping
import torch
import argparse
from threading import Thread
import time

FPS_SMOOTHNESS = 3

synth_map = dict()
buffers = dict()


def add_synth(synth_map, id, synth):
    if id not in synth_map:
        synth_map[id] = []
    synth_map[id].append(synth)


def face_params_changed(id, params):
    # synth = None
    #
    # if id not in synth_map:
    #     return
    # for s in synth_map[id]:
    #     if s.synthdef.name == 'grainer':
    #         synth = s
    #         break
    # if synth is None:
    #     return
    #
    # if params['emotion'] != 'neutral':
    #     buffer = synths.load_buffer_emotion(params['gender'], params['emotion'])
    # else:
    #     buffer = synths.load_buffer_class(0)
    # synth['buffer'] = buffer
    # if id in buffers:
    #     buffers[id].free()
    # buffers[id] = buffer
    pass


def change_params(synth, object):
    for m in param_mappings:
        m.apply(synth, object)


def update(synth, object):
    params_thread = Thread(target=change_params, args=(synth, object), daemon=True)
    params_thread.start()


def read():
    while True:
        available = tracker.read_video()
        fps = tracker.video_fps
        time.sleep(1 / (FPS_SMOOTHNESS * fps))
        if not available:
            time.sleep(0.01)


def display():
    while True:
        available = tracker.show_video()
        if not available:
            time.sleep(0.01)
        fps = tracker.video_fps
        time.sleep(1 / fps if fps else 0.002)


def process():
    while True:
        result = tracker.track()
        if result is None:
            time.sleep(0.01)
            continue

        all, new, deleted = result

        for o in new:
            # if o.class_id == 0:
            #     tracker.all_objs[o.id].params_changed = face_params_changed

            for m in synth_mappings:
                if o.class_id in m.object_classes:
                    s = synths.server.add_synth(m.synth_def, add_action='addBefore', fx_bus=fx_bus.bus_id)
                    if m.synth_def == synths.grainer:
                        buffers[o.id] = synths.load_buffer_class(o.class_id)
                        s['buffer'] = buffers[o.id]
                    add_synth(synth_map, o.id, s)

        for o in deleted:
            for s in synth_map[o.id]:
                s['gate'] = 0
            synth_map.pop(o.id)

        for o in all:
            if o.id in synth_map:
                for synth in synth_map[o.id]:
                    update(synth, o)


if __name__ == '__main__':
    print('cuda ' + ('not ' if not torch.cuda.is_available() else '') + 'available')

    parser = argparse.ArgumentParser(description='Video sonification based on object tracking')
    parser.add_argument('--source', type=str, default='0')
    opt = parser.parse_args()
    src = opt.source

    # src = 'test.mp4'
    # src = 'https://www.youtube.com/watch?v=b1LEJCV6kPc'
    # src = 'https://www.youtube.com/watch?v=WJLkXlhE1FM'
    # src = 'https://www.youtube.com/watch?v=HOASHDryAwU'
    # src = 'https://www.youtube.com/watch?v=gu5p_TdU9vw'

    tracker = Tracker(source=src)
    # tracker.classes = (tracker.get_class_index('person'))
    # tracker.classes = range(1, 80)

    fx_bus = synths.server.add_bus_group(2, 'audio')
    rev = synths.server.add_synth(synths.reverb, in_bus=fx_bus.bus_id)

    norm_x = lambda x: x / tracker.video_size[0]
    norm_y = lambda x: x / tracker.video_size[1]
    norm_area = lambda x: x / tracker.video_area
    norm_speed = lambda x: max(0, min(x / 50, 1.0))

    synth_mappings = list()
    param_mappings = list()

    synth_mappings.append(SynthMapping([0], synths.beeper))
    param_mappings.append(ParameterMapping('beeper', 'freq', 'y', scaling=lambda x: 20+800*(1-norm_y(x))))
    param_mappings.append(ParameterMapping('beeper', 'tempo', 'speed', scaling=lambda x: 30+100*norm_speed(x)))

    synth_mappings.append(SynthMapping(range(1, 80), synths.duster))
    param_mappings.append(ParameterMapping('duster', 'freq', 'y', scaling=lambda x: 50+5000*(1-norm_y(x))))
    param_mappings.append(ParameterMapping('duster', 'tempo', 'speed', scaling=lambda x: 100+100*norm_speed(x)))

    synth_mappings.append(SynthMapping(range(0, 80), synths.grainer))
    param_mappings.append(ParameterMapping('grainer', 'speed', 'speed', scaling=norm_speed))

    param_mappings.append(ParameterMapping(None, 'pan', 'x', scaling=lambda x: norm_x(x)*2-1))
    param_mappings.append(ParameterMapping(None, 'depth', 'area', scaling=lambda x: 0.6*(1-x/tracker.video_area)))
    param_mappings.append(ParameterMapping(None, 'level', 'area', scaling=lambda x: 0.2+0.5*(x/tracker.video_area)))

    # read, process, display
    read_thread = Thread(target=read)
    process_thread = Thread(target=process)
    display_thread = Thread(target=display)

    read_thread.start()
    process_thread.start()
    display_thread.start()

    read_thread.join()
    process_thread.join()
    display_thread.join()

    synths.server.quit()
