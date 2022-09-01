import supriya
from supriya.ugens import In, Out, SinOsc, Saw, Dust
from supriya.ugens.noise import WhiteNoise, Rand
from supriya.ugens.filters import BPF, Decay2
from supriya.ugens.panning import Pan2
from supriya.ugens.delay import AllpassC
from supriya.synthdefs import synthdef
import time

@synthdef()
def beeper(fx_bus, out_bus=0, direct=0.7, freq=1, tempo=120, level=0.1, pan=0):
    gen = SinOsc.ar(freq) * level * 0.5 * (Saw.kr(tempo / 60) + 0.5)
    Out.ar(out_bus, Pan2.ar(gen * direct, pan))
    Out.ar(fx_bus, Pan2.ar(gen * (1 - direct), pan))

@synthdef()
def popcorn(fx_bus, out_bus=0, direct=0.5, freq=50, tempo=120, level=0.5, pan=0):
    source = BPF.ar(level * WhiteNoise.ar(), freq, 1.0)
    gen = Decay2.ar(Dust.ar(tempo / 60), 0.01, 0.2) * source
    Out.ar(out_bus, Pan2.ar(gen * direct, pan))
    Out.ar(fx_bus, Pan2.ar(gen * (1 - direct), pan))

@synthdef()
def reverb(in_bus, out_bus=0):
    buf = In.ar(in_bus, 2)
    for i in range(16):
        buf = AllpassC.ar(buf, 0.04, [Rand(0.001, 0.04), Rand(0.001, 0.04)], 3)
    Out.ar(out_bus, buf)


if __name__ == '__main__':
    server = supriya.Server().boot()
    server.add_synthdef(reverb)
    server.add_synthdef(popcorn)
    server.add_synthdef(beeper)

    fx_bus = server.add_bus_group(2, 'audio')
    rev = server.add_synth(reverb, in_bus=fx_bus.bus_id)
    synth1 = server.add_synth(popcorn, add_action='addBefore', fx_bus=fx_bus.bus_id)
    synth2 = server.add_synth(beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
    time.sleep(1)
    synth1['freq'] = 800
    synth2['freq'] = 200
    time.sleep(10)
    synth1.release()
    synth2.release()
    server.quit()
