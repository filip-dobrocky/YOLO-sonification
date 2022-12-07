from deepface import DeepFace


class FaceTracker:
    def __init__(self):
        self.models = dict()
        self.models['emotion'] = DeepFace.build_model('Emotion')
        self.models['gender'] = DeepFace.build_model('Gender')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def get_face(self, img):
        face = DeepFace.analyze(img, actions=('emotion', 'gender'), models=self.models, enforce_detection=False)
        return face['gender'], face['dominant_emotion'], face['region']