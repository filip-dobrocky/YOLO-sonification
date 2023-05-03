import supriya
from supriya.ugens import In, Out, SinOsc, Saw, Dust, PlayBuf,\
                          FreeVerb, EnvGen, GrainBuf, Compander, Pulse, LFNoise2, Mix
from supriya.ugens.noise import WhiteNoise, Rand, ClipNoise
from supriya.ugens.filters import BPF, Decay2
from supriya.ugens.beq import BLowPass
from supriya.ugens.panning import Pan2
from supriya.ugens.delay import AllpassC
from supriya.synthdefs import synthdef, Envelope

import time
import os
import random

random.seed()
server = supriya.Server().boot(port=7400)

global_env = Envelope(
    amplitudes=(0, 1, 0),
    durations=(0.3, 0.3),
    curves=(2, -2),
    release_node=1
)


def load_buffer_class(cls):
    DIR = './audio/wav'
    files = [x for x in os.listdir(DIR) if x.startswith(str(cls)+'_')]
    f = random.choice(files)
    f = os.path.join(DIR, f)
    buffer = server.add_buffer(file_path=f, channel_count=1)
    buffer.normalize()
    return buffer


def load_buffer_emotion(gender, emotion):
    DIR = './audio/wav/emotions'
    files = [x for x in os.listdir(DIR) if x.startswith(gender+'_'+emotion)]
    f = random.choice(files)
    f = os.path.join(DIR, f)
    buffer = server.add_buffer(file_path=f, channel_count=1)
    buffer.normalize()
    return buffer


@synthdef()
def player(fx_bus, buffer, sample_rate, playback_rate=1, gate=1, out_bus=0, depth=0.4, level=0.5, pan=0):
    gen = PlayBuf.ar(
        buffer_id=buffer,
        rate=playback_rate*(sample_rate/server.status.actual_sample_rate),
        loop=True
    )
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    out_sig = envelope * level * Pan2.ar(gen, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def grainer(fx_bus, buffer, speed=0, speed_curve=0.6, gate=1, out_bus=0, depth=0.4, level=0.5, pan=0):
    dust = Dust.ar(3+(speed**speed_curve)*9)
    direction = ClipNoise.kr()
    gen = GrainBuf.ar(
        channel_count=1,
        buffer_id=buffer,
        rate=direction*(0.9+speed**speed_curve),
        duration=0.1+dust*1.5*(1-speed**speed_curve),
        position=dust,
        trigger=dust
    )
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    out_sig = envelope * level * Pan2.ar(Compander.ar(gen), pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def beeper(fx_bus, gate=1, out_bus=0, depth=0.4, freq=1, tempo=120, level=0.1, pan=0):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    gen = SinOsc.ar(freq) * level * 0.5 * (Saw.kr(tempo / 60) + 0.5)
    out_sig = 0.7 * envelope * Pan2.ar(gen, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def duster(fx_bus, gate=1, out_bus=0, depth=0.5, freq=50, tempo=120, level=0.5, pan=0):
    source = BPF.ar(level * WhiteNoise.ar(), freq, 1.0)
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    gen = Decay2.ar(Dust.ar(tempo / 60), 0.01, 0.2) * source
    out_sig = 0.9 * envelope * Pan2.ar(gen, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def droner(fx_bus, gate=1, out_bus=0, depth=0.2, freq=100, level=0.5, pan=0, movement=0.1, detune=0.05, cutoff=2000,
           resonance=0.0):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    lfo1 = LFNoise2.kr(0.1+movement*5)
    lfo2 = SinOsc.kr(movement+1+lfo1)
    lfo3 = SinOsc.kr(movement+1+lfo1, 1.57)
    n = 5
    oscillators = [(1/n)*Pulse.ar(freq + 0.1*(0.5+0.5*lfo1)*freq*detune*Rand.ir(-1.0, 1.0), 0.5 + 0.2*movement*lfo1)
                   for _ in range(n)]
    osc = Mix.new(oscillators)
    sub = SinOsc.ar(0.5 * freq)
    mix = 0.7 * (0.7 + 0.3 * movement * lfo2) * osc + (0.1 + 0.2 * (0.5*(1+lfo3))) * sub
    filter = BLowPass.ar(source=mix, frequency=cutoff, reciprocal_of_q=1-resonance)

    out_sig = 0.7 * envelope * Pan2.ar(filter, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def reverb(in_bus, out_bus=0):
    buf = In.ar(in_bus, 2)

    # for i in range(16):
    #     buf = AllpassC.ar(buf, 0.04, [Rand(0.001, 0.04), Rand(0.001, 0.04)], 3)

    buf = FreeVerb.ar(source=buf, mix=1.0, room_size=0.8, damping=0.4)
    Out.ar(out_bus, buf)


synthdefs = [reverb, duster, beeper, grainer, player, droner]
for s in synthdefs:
    server.add_synthdef(s)


if __name__ == '__main__':

    # b = load_buffer_class(0)
    b = load_buffer_emotion('Woman', 'sad')

    fx_bus = server.add_bus_group(2, 'audio')
    rev = server.add_synth(reverb, in_bus=fx_bus.bus_id)
    # synth1 = server.add_synth(popcorn, add_action='addBefore', fx_bus=fx_bus.bus_id)
    # synth2 = server.add_synth(beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
    # synth3 = server.add_synth(grainer, add_action='addBefore', fx_bus=fx_bus.bus_id, buffer=b, sample_rate=b.sample_rate, speed=0)
    synth3 = server.add_synth(droner, add_action='addBefore', fx_bus=fx_bus.bus_id)
    time.sleep(1)
    synth3['pan'] = 0.0
    synth3['depth'] = 0.1
    synth3['freq'] = 120
    # synth2['freq'] = 200
    # for p in synth1.synthdef.parameter_names:
    #     print(synth1[p])
    time.sleep(10)
    # synth1['gate'] = 0
    # synth2['gate'] = 0
    synth3['gate'] = 0
    time.sleep(4)
    server.quit()
