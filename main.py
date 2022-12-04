import supriya
import synths
from tracker import Tracker, Object
import torch


if __name__ == '__main__':
    print(torch.cuda.is_available())
    server = supriya.Server().boot()

    server.add_synthdef(synths.reverb)
    server.add_synthdef(synths.popcorn)
    server.add_synthdef(synths.beeper)

    fx_bus = server.add_bus_group(2, 'audio')
    rev = server.add_synth(synths.reverb, in_bus=fx_bus.bus_id)

    # tracker = Tracker(source='https://www.youtube.com/watch?v=b1LEJCV6kPc')
    tracker = Tracker(source="0")
    # tracker = Tracker(source='rtsp://:8555/stream')

    synth_map = dict()

    while True:
        try:
            all, new, deleted = tracker.track()
            for o in new:
                if o.class_id == 0:
                    synth_map[o.id] = server.add_synth(synths.beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
                else:
                    synth_map[o.id] = server.add_synth(synths.popcorn, add_action='addBefore', fx_bus=fx_bus.bus_id)
                # print("added synth")
            for o in deleted:
                synth_map[o.id].release()
                # print("released synth")
            for id, o in all.items():
                if id in synth_map:
                    synth = synth_map[id]
                    if synth.synthdef.name == 'beeper':
                        synth['freq'] = 20 + 2000 * (1 - o.pos[1] / tracker.video_size[1])
                        synth['pan'] = (o.pos[0] / tracker.video_size[0]) * 2 - 1
                        synth['tempo'] = 30 + o.speed * 50
                    elif synth.synthdef.name == 'popcorn':
                        synth['freq'] = 50 + 5000 * (1 - o.pos[1] / tracker.video_size[1])
                        synth['pan'] = (o.pos[0] / tracker.video_size[0]) * 2 - 1
                        synth['tempo'] = 100 + o.speed * 50
                    # print(f'set params {synth_map[id]["freq"]} {synth_map[id]["pan"]} {synth_map[id]["tempo"]}')

        except StopIteration:
            break

    server.quit()
