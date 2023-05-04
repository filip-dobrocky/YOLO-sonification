import musical_scales
import pitchtools
import bisect

scale_names = list(musical_scales.scale_intervals.keys())
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def expand_f_scale(freqs):
    min_f = 20
    max_f = 20_000

    expanded = freqs.copy()
    shifted = freqs

    while len(upper := [2*x for x in shifted if 2*x <= max_f]):
        expanded = expanded + upper
        shifted = upper
    shifted = freqs
    while len(lower := [x/2 for x in shifted if x/2 >= min_f]):
        expanded = lower + expanded
        shifted = lower

    return expanded


class Quantizer:
    def __init__(self, root='C', scale='ionian', ref_freq=440):
        self.root = root
        self.ref_freq = ref_freq
        self.scale_name = scale

        notes = musical_scales.scale(root, scale)
        cnv = pitchtools.PitchConverter(a4=ref_freq)
        self.freqs = expand_f_scale([cnv.n2f(n.midi) for n in notes])

    def snap(self, freq):
        i = bisect.bisect_left(self.freqs, freq)
        if i >= len(self.freqs):
            i = len(self.freqs) - 1
        elif i and self.freqs[i] - freq > freq - self.freqs[i - 1]:
            i = i - 1
        return self.freqs[i]
