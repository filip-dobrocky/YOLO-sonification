from deepface import DeepFace

models = dict()
models['emotion'] = DeepFace.build_model('Emotion')
models['gender'] = DeepFace.build_model('Gender')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def get_face(img):
    face = DeepFace.analyze(img, actions=('emotion', 'gender'), models=models, enforce_detection=False)
    return face['gender'], face['dominant_emotion'], face['region']
