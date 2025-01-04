from .__utils import detect_faces
from torch import Tensor


def assess_image(image: Tensor):
    face_detections = detect_faces(image)

    assessment_parameters = []

    for face_detection in face_detections:
        confidence = face_detection["confidence"]

        assessment_parameters.append(confidence)

    return len(face_detections), assessment_parameters
