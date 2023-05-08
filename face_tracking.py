from deepface import DeepFace

gender_labels = ['Man', 'Woman']
emotion_labels = ['neutral', 'happy', 'surprise', 'fear', 'sad', 'angry', 'disgust']


def get_face(img):
    face = DeepFace.analyze(img, actions=('emotion', 'gender'), enforce_detection=False, silent=True)[0]
    return face['dominant_gender'], face['dominant_emotion'], face['region'], face['emotion'][face['dominant_emotion']]
