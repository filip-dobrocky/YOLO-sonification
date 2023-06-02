import supriya
from supriya.ugens import In, Out, SinOsc, Saw, Dust, PlayBuf, FreeVerb, EnvGen, GrainBuf, Compander, Pulse,\
                          LFNoise2, Mix, Limiter, LinExp, LinLin, LFSaw, DC, Hasher, Sweep, Linen, Impulse, \
                          UnaryOpUGen, Clip
from supriya.ugens.noise import WhiteNoise, Rand, ClipNoise, CoinGate, LFClipNoise
from supriya.ugens.filters import BPF, Decay2, LPF
from supriya.ugens.beq import BLowPass
from supriya.ugens.panning import Pan2
from supriya.ugens.delay import AllpassC
from supriya.synthdefs import synthdef, Envelope

import time
import os
import random
import math
import logging
from threading import Thread

random.seed()
server = supriya.Server().boot(port=7400, buffer_count=2048)

global_env = Envelope(
    amplitudes=(0, 1, 0),
    durations=(0.3, 0.3),
    curves=(2, -2),
    release_node=1
)

BUFFERS = {}


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


def preload_buffers():
    CLASS_DIR = './audio/wav'
    class_files = os.listdir(CLASS_DIR)
    for f in class_files:
        if not f.endswith('.wav'):
            continue
        file = os.path.join(CLASS_DIR, f)
        cls = f.split('_')[0]
        if cls not in BUFFERS:
            BUFFERS[cls] = []
        buf = server.add_buffer(file_path=file, channel_count=1)
        # buf.normalize()
        BUFFERS[cls].append(buf)

    EMO_DIR = './audio/wav/emotions'
    emo_files = os.listdir(EMO_DIR)
    for f in emo_files:
        if not f.endswith('.wav'):
            continue
        file = os.path.join(EMO_DIR, f)
        name = f.split('_')
        sex = name[0]
        emotion = name[1]
        if sex not in BUFFERS:
            BUFFERS[sex] = {}
        if emotion not in BUFFERS[sex]:
            BUFFERS[sex][emotion] = []
        buf = server.add_buffer(file_path=file, channel_count=1)
        # buf.normalize()
        BUFFERS[sex][emotion].append(buf)
    logging.info('Buffers loaded.')


def random_class_buffer(class_id):
    try:
        buf = random.choice(BUFFERS[str(class_id)])
        return buf.buffer_id, buf.sample_rate
    except KeyError:
        return 0, 0


def random_emotion_buffer(sex, emotion):
    try:
        if emotion != 'neutral':
            buf = random.choice(BUFFERS[sex][emotion])
            return buf.buffer_id, buf.sample_rate
        return random_class_buffer(0)
    except KeyError:
        return 0, 0


def random_sex_buffer(sex):
    try:
        buf = random.choice(random.choice(BUFFERS[sex]))
        return buf.buffer_id, buf.sample_rate
    except KeyError:
        return 0, 0


@synthdef()
def player(fx_bus, sample_rate, gate=1, out_bus=0, level=0.5, pan=0, depth=0.4, buffer_id=0, playback_rate=1):
    gen = PlayBuf.ar(
        buffer_id=buffer_id,
        rate=playback_rate*(sample_rate/server.status.actual_sample_rate),
        loop=True
    )
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    out_sig = envelope * level * Pan2.ar(gen, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def grainer(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.4, buffer_id=0, grain_duration=50, scan=1.0,
            randomize=0.5, density=0.1):
    dur_s = grain_duration.clip(1, 200) / 1000
    dust = Dust.ar(density.clip(0.05, 1) * 20 / dur_s)
    dur_offset = randomize * 0.01 * dust
    direction = 1 - 2 * CoinGate.ar(randomize, dust)

    gen = GrainBuf.ar(
        channel_count=1,
        buffer_id=buffer_id,
        rate=direction * scan,
        duration=dur_s + dur_offset,
        position=dust * randomize,
        trigger=dust > 0
    )
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    out_sig = envelope * level * Pan2.ar(Compander.ar(gen), pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def beeper(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.4, freq=1, tempo=120):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    gen = SinOsc.ar(freq) * level * 0.5 * (Saw.kr(tempo / 60) + 0.5)
    out_sig = envelope * Pan2.ar(gen, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def duster(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.5, freq=50, tempo=120):
    source = BPF.ar(level * WhiteNoise.ar(), freq, 0.5)
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    gen = Decay2.ar(Dust.ar(tempo / 60), 0.01, 0.2) * source
    out_sig = envelope * Pan2.ar(gen, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def droner(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.2, freq=100, movement=0.1, detune=0.05, f_cutoff=2000,
           f_resonance=0.0):
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
    filter = BLowPass.ar(source=mix, frequency=f_cutoff, reciprocal_of_q=1-f_resonance.clip(0, 1))

    out_sig = 0.7 * envelope * Pan2.ar(filter, pan) * level
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def operator(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.4, freq=100, tempo=30,
             fm_ratio1=0.3, fm_ratio2=0.3, fm_depth1=200, fm_depth2=0, env_curve1=0.5, env_curve2=2):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    tempo_env = 0.5 * (Saw.kr(tempo / 60) + 0.5)
    mod2 = SinOsc.ar(fm_ratio2*freq) * tempo_env ** env_curve2
    mod1 = SinOsc.ar(fm_ratio1*freq + fm_depth2*mod2) * tempo_env ** env_curve1
    carrier = SinOsc.ar(freq + fm_depth1*mod1) * level * tempo_env
    out_sig = 0.7 * envelope * Pan2.ar(carrier, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def pulsar(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.4, freq=1, formant_freq=50, sine_cycles=2,
           window_curve=2):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    phase = LinLin.ar(LFSaw.ar(freq, 1), -1, 1, 0, 1) * formant_freq / freq
    window = LinLin.ar(phase, 0, 1, 1, 0) ** window_curve
    sine = UnaryOpUGen(source=phase*2*math.pi*sine_cycles.floor(),
                       special_index=supriya.UnaryOperator.SIN,
                       calculation_rate=supriya.CalculationRate.AUDIO)
    sig = sine * window * (phase < 1)
    out_sig = level * envelope * Pan2.ar(sig, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def drummer(fx_bus, gate=1, out_bus=0, level=0.5, pan=0, depth=0.4, tempo=100, density=0.7,
            vol_kick=1.0, vol_snare=1.0, vol_hat=1.0, vol_clap=1.0, f_cutoff=15000, f_resonance=0.0):
    envelope = EnvGen.kr(envelope=global_env, gate=gate, done_action=2)
    pulse = Impulse.ar(tempo/60)

    kick_gate = CoinGate.ar(density, pulse)
    kick = SinOsc.ar(EnvGen.ar(envelope=Envelope((200, 50, 50), (0.05, 0.2)), gate=kick_gate))
    kick = kick + (Hasher.ar(Sweep.ar()) * EnvGen.ar(envelope=Envelope.percussive(0.001, 0.001, 0.001), gate=kick_gate))
    kick = kick * EnvGen.ar(envelope=Envelope.percussive(0.001, 1.0), gate=kick_gate)

    snare_gate = CoinGate.ar(density*0.5, pulse)
    snare = SinOsc.ar(EnvGen.ar(envelope=Envelope((800, 210, 200), (0.01, 0.2)), gate=snare_gate))
    snare = snare + (BPF.ar(Hasher.ar(Sweep.ar()), 3000, 0.7) *
                     EnvGen.ar(envelope=Envelope.percussive(0.1, 0.3, 5), gate=snare_gate))
    snare = snare.tanh() * EnvGen.ar(envelope=Envelope.percussive(0.001, 0.3), gate=snare_gate)

    hat_gate = CoinGate.ar(density*0.8, pulse)
    hat = BPF.ar(Hasher.ar(Sweep.ar()), 9000, 0.1) * 6
    hat = hat * EnvGen.ar(envelope=Envelope.percussive(0.02, 0.03), gate=hat_gate)

    clap_gate = CoinGate.ar(density*0.3, pulse)
    clap = BPF.ar(Hasher.ar(Sweep.ar()), 1100, 0.3) * 8
    clap = clap * EnvGen.ar(envelope=Envelope((0, 1, 0, 1, 0, 1, 0), (0.001, 0.01, 0.001, 0.01, 0.001, 0.08)),
                            gate=clap_gate)

    sig = vol_kick.clip(0, 1) * 2 * kick \
        + vol_snare.clip(0, 1) * 0.8 * snare \
        + vol_hat.clip(0, 1) * 0.9 * Pan2.ar(hat, 0.45) \
        + vol_clap.clip(0, 1) * Pan2.ar(clap, -0.15)
    filter = BLowPass.ar(source=sig, frequency=f_cutoff, reciprocal_of_q=1-f_resonance.clip(0, 0.9))
    out_sig = level * envelope * Pan2.ar(filter, pan)
    Out.ar(fx_bus, out_sig * depth)
    Out.ar(out_bus, out_sig * (1 - depth))


@synthdef()
def reverb(in_bus, out_bus=0):
    buf = In.ar(in_bus, 2)
    buf = FreeVerb.ar(source=buf, mix=1.0, room_size=0.8, damping=0.4)
    Out.ar(out_bus, buf)


@synthdef()
def limiter(in_bus, level=1.0, duration=0.01, out_bus=0):
    buf = In.ar(in_bus, 2)

    buf = Limiter.ar(source=buf, level=level, duration=duration)
    Out.ar(out_bus, buf)


synthdefs = [duster, beeper, grainer, player, droner, operator, pulsar, drummer]
for s in synthdefs:
    server.add_synthdef(s)

server.add_synthdef(reverb)
server.add_synthdef(limiter)

buffers_thread = Thread(target=preload_buffers, daemon=True)
buffers_thread.start()


if __name__ == '__main__':
    # b = load_buffer_class(0)
    b1 = load_buffer_emotion('Woman', 'sad')
    b2 = load_buffer_emotion('Man', 'angry')

    fx_bus = server.add_bus_group(2, 'audio')
    rev = server.add_synth(reverb, in_bus=fx_bus.bus_id)
    # synth1 = server.add_synth(popcorn, add_action='addBefore', fx_bus=fx_bus.bus_id)
    # synth2 = server.add_synth(beeper, add_action='addBefore', fx_bus=fx_bus.bus_id)
    # synth3 = server.add_synth(grainer, add_action='addBefore', fx_bus=fx_bus.bus_id, buffer=b2.buffer_id, speed=0)
    # synth3 = server.add_synth(droner, add_action='addBefore', fx_bus=fx_bus.bus_id)
    synth3 = server.add_synth(pulsar, add_action='addBefore', fx_bus=fx_bus.bus_id)
    # synth3 = server.add_synth(drummer, add_action='addBefore', fx_bus=fx_bus.bus_id)
    time.sleep(1)
    # synth3['pan'] = 1.0
    synth3['level'] = 1.0
    # synth3['depth'] = 0.1
    # synth3['tempo'] = 300
    synth3['freq'] = 5
    synth3['formant_freq'] = 80
    # for p in synth1.synthdef.parameter_names:
    #     print(synth1[p])
    time.sleep(3)
    synth3['sine_cycles'] = 5.1
    # synth3['tempo'] = 800
    # synth3['pan'] = -1.0
    # synth3['buffer_id'] = b1.buffer_id
    # synth1['gate'] = 0
    # synth2['gate'] = 0
    time.sleep(4)
    synth3['sine_cycles'] = 8.8
    time.sleep(4)
    synth3['gate'] = 0
    time.sleep(4)
    server.quit()
