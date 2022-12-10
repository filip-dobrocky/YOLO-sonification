import supriya
import synths
from tracker import Tracker, Object
import torch

synth_map = dict()
buffers = dict()


def add_synth(synth_map, id, synth):
    if id not in synth_map:
        synth_map[id] = []
    synth_map[id].append(synth)


def face_params_changed(id, params):
    synth = None
    buffer = None
    for s in synth_map[id]:
        if s.synthdef.name == 'grainer':
            synth = s
            break
    if synth is not None:
        synth['gate'] = 0
        if params['emotion'] != 'neutral':
            buffer = synths.load_buffer_emotion(params['gender'], params['emotion'])
        else:
            buffer = synths.load_buffer_class(0)
        new_synth = synths.server.add_synth(synths.grainer,
                                            add_action='addBefore', fx_bus=fx_bus.bus_id,
                                            buffer=buffer)
        add_synth(synth_map, id, new_synth)
    synth_map[id].remove(synth)
    buffers[id] = buffer


if __name__ == '__main__':
    print('cuda ' + ('not ' if not torch.cuda.is_available() else '') + 'available' )

    fx_bus = synths.server.add_bus_group(2, 'audio')
    rev = synths.server.add_synth(synths.reverb, in_bus=fx_bus.bus_id)

    # tracker = Tracker(source='https://www.youtube.com/watch?v=b1LEJCV6kPc')
    # tracker = Tracker(source='https://www.youtube.com/watch?v=WJLkXlhE1FM')
    # tracker = Tracker(source='https://www.youtube.com/watch?v=HOASHDryAwU')
    # tracker = Tracker(source='https://www.youtube.com/watch?v=gu5p_TdU9vw')
    tracker = Tracker(source="0")
    # tracker = Tracker(source="D:\\Videos\\_VCR\\14 (5_2002-1_2003)_Trim.mp4")
    # tracker = Tracker(source='rtsp://:8555/stream')

    norm_h_pos = lambda x: x / tracker.video_size[0]
    norm_v_pos = lambda x: x / tracker.video_size[1]
    norm_area = lambda x: x / tracker.video_area
    norm_speed = lambda x: max(0, min(x / 50, 1.0))

    while True:
        try:
            all, new, deleted = tracker.track()

            for o in new:
                if o.class_id == 0:
                    tracker.all_objs[o.id].params_changed = face_params_changed
                    s = synths.server.add_synth(synths.beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
                    add_synth(synth_map, o.id, s)
                else:
                    s = synths.server.add_synth(synths.duster, add_action='addBefore', fx_bus=fx_bus.bus_id)
                    add_synth(synth_map, o.id, s)
                buffers[o.id] = synths.load_buffer_class(o.class_id)
                s = synths.server.add_synth(synths.grainer,
                                            add_action='addBefore', fx_bus=fx_bus.bus_id,
                                            buffer=buffers[o.id])
                add_synth(synth_map, o.id, s)

            for o in deleted:
                for s in synth_map[o.id]:
                    s['gate'] = 0
                synth_map.pop(o.id)

            for o in all:
                if o.id in synth_map:
                    for synth in synth_map[o.id]:
                        if synth.synthdef.name == 'beeper':
                            synth['freq'] = 20 + 800 * (1 - norm_v_pos(o.pos[1]))
                            synth['tempo'] = 30 + norm_speed(o.speed) * 100
                        elif synth.synthdef.name == 'duster':
                            synth['freq'] = 50 + 5000 * (1 - norm_v_pos(o.pos[1]))
                            synth['tempo'] = 100 + norm_speed(o.speed) * 100
                        elif synth.synthdef.name == 'grainer':
                            synth['speed'] = norm_speed(o.speed)
                        synth['pan'] = norm_h_pos(o.pos[0]) * 2 - 1
                        synth['depth'] = (1 - (o.area / tracker.video_area)) * 0.6
                        synth['level'] = 0.2 + 0.5 * (o.area / tracker.video_area)

        except StopIteration:
            break

    synths.server.quit()
