from deepface import DeepFace

models = dict()
models['emotion'] = DeepFace.build_model('Emotion')
models['gender'] = DeepFace.build_model('Gender')
gender_labels = ['Man', 'Woman']
emotion_labels = ['neutral', 'happy', 'surprise', 'fear', 'sad', 'angry', 'disgust']


def get_face(img):
    face = DeepFace.analyze(img, actions=('emotion', 'gender'), models=models, enforce_detection=False)
    return face['gender'], face['dominant_emotion'], face['region'], face['emotion'][face['dominant_emotion']]
