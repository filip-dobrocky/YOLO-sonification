import logging

from deepface import DeepFace
import os
import sys

gender_labels = ['Man', 'Woman']
emotion_labels = ['neutral', 'happy', 'surprise', 'fear', 'sad', 'angry', 'disgust']


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        pass


def get_face(img):
    with HiddenPrints():
        try:
            face = DeepFace.analyze(img, actions=('emotion', 'gender'), enforce_detection=False, silent=True,
                                    detector_backend='mtcnn')[0]
        except Exception as e:
            logging.warning('Face analysis failed.')
            return None
    return face['dominant_gender'], face['dominant_emotion'], face['region'], face['emotion'][face['dominant_emotion']]
