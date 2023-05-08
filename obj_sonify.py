import supriya
import synths
from tracker import Tracker, Object
import mapping
from mapping import SynthMapping, ParameterMapping, mapping_applies, synth_mappings, param_mappings
import gui


import torch
import argparse
from threading import Thread
import time
import logging

FPS_SMOOTHNESS = 1.2

synth_map = dict()
buffers = dict()

logging.disable(logging.DEBUG)
supriya.osc.osc_protocol_logger.setLevel(logging.WARNING)

threads_running = True


def add_synth(id, class_id, synthdef):
    s = synths.server.add_synth(synthdef, add_action=supriya.AddAction.ADD_BEFORE, fx_bus=fx_bus.bus_id,
                                out_bus=out_bus.bus_id)
    if synthdef == synths.grainer:
        buf_id, _ = synths.random_class_buffer(class_id)
        s['buffer_id'] = buf_id
    elif synthdef == synths.player:
        buf_id, sample_rate = synths.random_class_buffer(class_id)
        s['buffer_id'] = buf_id
        s['sample_rate'] = sample_rate

    if id not in synth_map:
        synth_map[id] = []
    synth_map[id].append(s)


def change_params(synth, object):
    for m in param_mappings:
        if mapping_applies(m, synth):
            m.apply(synth, object)


def update(synth, object):
    params_thread = Thread(target=change_params, args=(synth, object), daemon=True)
    params_thread.start()


def read():
    while threads_running:
        available = tracker.running and tracker.read_video()
        if not available:
            time.sleep(0.01)
            continue
        fps = tracker.video_fps
        time.sleep(1 / (FPS_SMOOTHNESS * fps) if fps > 0 else 0.002)


def display():
    while threads_running:
        available = tracker.running and tracker.show_video()
        if not available:
            time.sleep(0.01)
            continue
        fps = tracker.video_fps
        time.sleep(1 / fps if fps > 0 else 0.002)


def process():
    while threads_running:
        if not tracker.running:
            time.sleep(0.01)
            continue

        result = tracker.track()
        if result is None:
            time.sleep(0.01)
            continue

        all, new, deleted = result

        for o in new:
            for m in synth_mappings:
                if o.class_id in m.object_classes:
                    add_synth(o.id, o.class_id, m.synthdef)

        for o in deleted:
            if o.id not in synth_map:
                break
            for s in synth_map[o.id]:
                s['gate'] = 0
            synth_map.pop(o.id)

        for o in all:
            if o.id in synth_map:
                for s in synth_map[o.id].copy():
                    if any([o.class_id in m.object_classes and m.synthdef == s.synthdef for m in synth_mappings]):
                        update(s, o)
                    else:
                        s['gate'] = 0
                        synth_map[o.id].remove(s)

            for m in synth_mappings:
                if o.class_id in m.object_classes:
                    if o.id not in synth_map or not any([s.synthdef.name == m.synthdef.name for s in synth_map[o.id]]):
                        add_synth(o.id, o.class_id, m.synthdef)


if __name__ == '__main__':
    logging.info('Cuda ' + ('not ' if not torch.cuda.is_available() else '') + 'available.')

    parser = argparse.ArgumentParser(description='Video sonification based on object tracking')
    parser.add_argument('--buffer', type=int, default='3')
    opt = parser.parse_args()
    buf_len = opt.buffer

    tracker = Tracker(buf_len=buf_len)

    fx_bus = synths.server.add_bus_group(2, 'audio')
    out_bus = synths.server.add_bus_group(2, 'audio')
    rev = synths.server.add_synth(synths.reverb, in_bus=fx_bus.bus_id, out_bus=out_bus.bus_id)
    limiter = synths.server.add_synth(synths.limiter, level=1.0, duration=0.01,
                                      in_bus=out_bus.bus_id,
                                      add_action=supriya.AddAction.ADD_TO_TAIL)

    mapping.norm_x = lambda x: x / tracker.video_size[0]
    mapping.norm_y = lambda x: x / tracker.video_size[1]
    mapping.norm_area = lambda x: x / tracker.video_area

    # read, process, display
    read_thread = Thread(target=read)
    process_thread = Thread(target=process)
    display_thread = Thread(target=display)

    read_thread.start()
    process_thread.start()
    display_thread.start()

    gui.tracker = tracker
    gui.run()

    threads_running = False
    read_thread.join(1)
    process_thread.join(1)
    display_thread.join(1)
    synths.server.quit()
