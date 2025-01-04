from torch import Tensor
from mtcnn import MTCNN
import einops
import numpy as np
import torch
import warnings

detector = MTCNN()


def tensor2cv(tensor: Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = einops.rearrange(tensor, '1 c h w -> c h w')

    return einops.rearrange(tensor, 'c h w -> h w c').numpy()


def cv2tensor(cv: np.ndarray) -> Tensor:
    return torch.from_numpy(einops.rearrange(cv, 'h w c -> c h w'))


def detect_faces(image):
    assert isinstance(image, Tensor), 'Should pass a tensor.'
    assert image.ndim in [3, 4], 'Should pass a tensor with 3 or 4 dims: (batch), color, height, width.'

    assert image.shape[0] == 3 or image.shape[1] == 3, 'Should pass an RGB image with dims: (batch), color, height, width.'

    detection_ready_image = tensor2cv(image) * 255

    return detector.detect_faces(detection_ready_image)


def detect_face(image, *, warn=True):
    face_detections = detect_faces(image)

    n_faces = len(face_detections)

    if n_faces == 0:
        raise ValueError("No faces detected.")

    if n_faces != 1:
        if warn:
            warnings.warn(f"Found {n_faces} faces. Aligning to the one with biggest confidence.")
        face_detection = max(face_detections, key=lambda fd: fd['confidence'])
    else:
        face_detection = face_detections[0]

    return face_detection


