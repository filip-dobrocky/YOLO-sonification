import supriya
from supriya.ugens import In, Out, SinOsc, Saw, Dust, PlayBuf, FreeVerb, EnvGen
from supriya.ugens.noise import WhiteNoise, Rand
from supriya.ugens.filters import BPF, Decay2
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
    durations=(1, 1),
    curves=(2, -2),
    release_node=1
)
# global_env = Envelope.asr(attack_time=0.5, sustain=1.0, release_time=0.5)

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
def beeper(fx_bus, gate=1, out_bus=0, depth=0.4, freq=1, tempo=120, level=0.1, pan=0):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    gen = SinOsc.ar(freq) * level * 0.5 * (Saw.kr(tempo / 60) + 0.5)
    panned = envelope * Pan2.ar(gen, pan)
    Out.ar(fx_bus, panned * depth)
    Out.ar(out_bus, panned * (1 - depth))


@synthdef()
def popcorn(fx_bus, gate=1, out_bus=0, depth=0.5, freq=50, tempo=120, level=0.5, pan=0):
    source = BPF.ar(level * WhiteNoise.ar(), freq, 1.0)
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    gen = Decay2.ar(Dust.ar(tempo / 60), 0.01, 0.2) * source
    panned = envelope * Pan2.ar(gen, pan)
    Out.ar(fx_bus, panned * depth)
    Out.ar(out_bus, panned * (1 - depth))


@synthdef()
def player(fx_bus, buffer, sample_rate, playback_rate, gate=1, out_bus=0, depth=0.4, level=0.5, pan=0):
    gen = PlayBuf.ar(
        buffer_id=buffer,
        rate=playback_rate*(sample_rate/server.status.actual_sample_rate),
        loop=True
    )
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    panned = envelope * level * Pan2.ar(gen, pan)
    Out.ar(fx_bus, panned * depth)
    Out.ar(out_bus, panned * (1 - depth))


@synthdef()
def reverb(in_bus, out_bus=0):
    buf = In.ar(in_bus, 2)

    # for i in range(16):
    #     buf = AllpassC.ar(buf, 0.04, [Rand(0.001, 0.04), Rand(0.001, 0.04)], 3)

    buf = FreeVerb.ar(source=buf, mix=1.0, room_size=0.7)
    Out.ar(out_bus, buf)


if __name__ == '__main__':
    server = supriya.Server().boot()
    server.add_synthdef(reverb)
    server.add_synthdef(popcorn)
    server.add_synthdef(beeper)

    b = load_buffer(server, '0', 0)

    fx_bus = server.add_bus_group(2, 'audio')
    rev = server.add_synth(reverb, in_bus=fx_bus.bus_id)
    synth1 = server.add_synth(popcorn, add_action='addBefore', fx_bus=fx_bus.bus_id)
    synth2 = server.add_synth(beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
    synth3 = server.add_synth(player, add_action='addBefore', fx_bus=fx_bus.bus_id, buffer=b, sample_rate=b.sample_rate)
    time.sleep(1)
    synth1['freq'] = 800
    synth2['freq'] = 200
    for p in synth1.synthdef.parameter_names:
        print(synth1[p])
    time.sleep(10)
    synth1.release()
    synth2.release()
    server.quit()
