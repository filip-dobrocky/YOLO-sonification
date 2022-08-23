import supriya
import synths
from tracker import Tracker, Object


if __name__ == '__main__':
    server = supriya.Server().boot()
    server.add_synthdef(synths.beeper)

    tracker = Tracker(source='https://www.youtube.com/watch?v=MtwY9xKfaYo')

    synth_map = dict()

    while True:
        try:
            tracker.track()
            for o in tracker.new_objs:
                synth_map[o.id] = server.add_synth(synths.beeper)
                print("added synth")
            for o in tracker.del_objs:
                synth_map[o.id].release()
                print("released synth")
            for id, o in tracker.all_objs.items():
                if id in synth_map:
                    synth_map[id]['freq'] = 50 + 1000 * (1 - o.pos[1] / tracker.video_size[1])
                    synth_map[id]['pan'] = (o.pos[0] / tracker.video_size[0]) * 2 - 1
                    synth_map[id]['tempo'] = 30 + o.speed * 100
                    print(f'set params {synth_map[id]["freq"]} {synth_map[id]["pan"]} {synth_map[id]["tempo"]}')

        except StopIteration:
            print(f'Done. ({time.time() - t0:.3f}s)')
            break

    server.quit()