from .__utils import detect_faces, find_pose
from torch import Tensor


def assess_quality(image: Tensor):
    face_detections = detect_faces(image)

    assessment_parameters = []

    for face_detection in face_detections:
        confidence = face_detection["confidence"]
        head_position = find_pose(face_detection["keypoints"])

        assessment_parameters.append({'confidence': confidence, **head_position})

    return len(face_detections), assessment_parameters
