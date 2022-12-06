import freesound
from tracker import Tracker
from os import path
from pydub import AudioSegment

client = freesound.FreesoundClient()
client.set_token("6rsKbmyWUy0hCsrD59v2JAZrAA3MpgDrm9TDzaT8", "token")

tracker = Tracker(source='0')

for i, name in tracker.names.items():
    results = client.text_search(
        query=name,
        filter="duration:[1.0 TO 15.0]",
        fields="id,name,previews"
    )

    for j, sound in enumerate(results):
        sound.retrieve_preview("./mp3", str(i) + "_" + name.replace(" ", "-") + "_" + str(j) + ".mp3")
        print(sound.name)
        if j >= 20:
            break
