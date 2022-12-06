import supriya
import synths
from tracker import Tracker, Object
import torch


def add_synth(synth_map, id, synth):
    if id not in synth_map:
        synth_map[id] = []
    synth_map[id].append(synth)


if __name__ == '__main__':
    print(torch.cuda.is_available())

    synths.server.add_synthdef(synths.reverb)
    synths.server.add_synthdef(synths.popcorn)
    synths.server.add_synthdef(synths.beeper)

    fx_bus = synths.server.add_bus_group(2, 'audio')
    rev = synths.server.add_synth(synths.reverb, in_bus=fx_bus.bus_id)

    # tracker = Tracker(source='https://www.youtube.com/watch?v=b1LEJCV6kPc')
    tracker = Tracker(source='https://www.youtube.com/watch?v=HOASHDryAwU')
    # tracker = Tracker(source="0")
    # tracker = Tracker(source="D:\\Videos\\_VCR\\14 (5_2002-1_2003)_Trim.mp4")
    # tracker = Tracker(source='rtsp://:8555/stream')

    synth_map = dict()
    buffers = dict()

    while True:
        try:
            all, new, deleted = tracker.track()

            for o in new:
                if o.class_id == 0:
                    s = synths.server.add_synth(synths.beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
                    add_synth(synth_map, o.id, s)
                else:
                    s = synths.server.add_synth(synths.popcorn, add_action='addBefore', fx_bus=fx_bus.bus_id)
                    add_synth(synth_map, o.id, s)
                buffers[o.id] = synths.load_buffer(synths.server, o.id, o.class_id)
                s = synths.server.add_synth(synths.player,
                                            add_action='addBefore', fx_bus=fx_bus.bus_id,
                                            buffer=buffers[o.id], sample_rate=buffers[o.id].sample_rate)
                add_synth(synth_map, o.id, s)
            for o in deleted:
                if o.id in buffers:
                    buffers[o.id].free()
                for s in synth_map[o.id]:
                    s.release()
                synth_map.pop(o.id)
            for o in all:
                if o.id in synth_map:
                    for synth in synth_map[o.id]:
                        if synth.synthdef.name == 'beeper':
                            synth['freq'] = 20 + 2000 * (1 - o.pos[1] / tracker.video_size[1])
                            synth['pan'] = (o.pos[0] / tracker.video_size[0]) * 2 - 1
                            synth['tempo'] = 30 + o.speed * 50
                        elif synth.synthdef.name == 'popcorn':
                            synth['freq'] = 50 + 5000 * (1 - o.pos[1] / tracker.video_size[1])
                            synth['pan'] = (o.pos[0] / tracker.video_size[0]) * 2 - 1
                            synth['tempo'] = 100 + o.speed * 50
                        elif synth.synthdef.name == 'player':
                            synth['pan'] = (o.pos[0] / tracker.video_size[0]) * 2 - 1
                            synth['playback_rate'] = 1.0 + o.speed / 50

                        synth['depth'] = (1 - (o.area / tracker.video_area)) * 0.6
                        synth['level'] = 0.2 + 0.5 * (o.area / tracker.video_area)

        except StopIteration:
            break

    synths.server.quit()
