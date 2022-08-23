import supriya
from supriya.ugens import Out, SinOsc, Saw
from supriya.ugens.panning import Pan2
from supriya.synthdefs import synthdef
import time

@synthdef()
def beeper(freq = 440, tempo = 120, level = 0.1, pan = 0):
    gen = SinOsc.ar(freq) * level * 0.5 * (Saw.kr(tempo / 60) + 0.5)
    Out.ar(0, Pan2.ar(gen, pan))

if __name__ == '__main__':
    server = supriya.Server().boot()
    server.add_synthdef(beeper)
    synth = server.add_synth(beeper)
    time.sleep(10)
    synth.release()
    server.quit()