import os
from pydub import AudioSegment, effects


DIR1 = './wav'
DIR2 = './wav/emotions'
directory = os.listdir(DIR1)
for f in directory:
    if not f.endswith('.wav'):
        continue
    src = os.path.join(DIR1, f)
    dst = os.path.join(DIR1+'/normalized', f)
    sound = AudioSegment.from_file(src)
    sound = effects.normalize(sound)
    sound.export(dst, format="wav")

directory = os.listdir(DIR2)
for f in directory:
    if not f.endswith('.wav'):
        continue
    src = os.path.join(DIR2, f)
    dst = os.path.join(DIR2+'/normalized', f)
    sound = AudioSegment.from_file(src)
    sound = effects.normalize(sound)
    sound.export(dst, format="wav")
