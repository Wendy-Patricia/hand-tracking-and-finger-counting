"""Traitement d'une image, utilisable sans caméra ni fenêtre OpenCV."""

from dataclasses import dataclass

import numpy as np

from config import ProcessingParameters
from hand_detection import HandDetection, detect_hand
from segmentation import segment_hand


@dataclass
class FrameResult:
    mask: np.ndarray
    detection: HandDetection | None


def process_frame(frame: np.ndarray, parameters: ProcessingParameters) -> FrameResult:
    """Segmente puis analyse une image BGR sans modifier l'image source."""
    _, mask = segment_hand(frame, parameters)
    return FrameResult(mask, detect_hand(mask, parameters.min_area))
