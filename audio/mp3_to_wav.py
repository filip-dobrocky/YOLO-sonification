import os
from pydub import AudioSegment


DIR = './mp3'
directory = os.listdir(DIR)
for f in directory:
    src = os.path.join(DIR, f)
    dst = src.replace('mp3', 'wav')
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
