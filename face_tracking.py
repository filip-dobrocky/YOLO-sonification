from deepface import DeepFace


class FaceTracker:
    def __init__(self):
        self.models = dict()
        self.models['emotion'] = DeepFace.build_model('Emotion')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def get_emotion(self, img):
        face = DeepFace.analyze(img, actions=['emotion'], models=self.models, enforce_detection=False)
        return face['dominant_emotion'], face['region']