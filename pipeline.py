"""Traitement d'une image, utilisable sans caméra ni fenêtre OpenCV."""

from dataclasses import dataclass

import numpy as np

from config import ProcessingParameters
from hand_detection import HandDetection, detect_hand
from hand_geometry import analyze_fingers
from segmentation import segment_hand


@dataclass
class FrameResult:
    mask: np.ndarray
    detection: HandDetection | None
    analysis: dict


def process_frame(frame: np.ndarray, parameters: ProcessingParameters) -> FrameResult:
    """Segmente puis analyse une image BGR sans modifier l'image source."""
    _, mask = segment_hand(frame, parameters)
    detection = detect_hand(mask, parameters.min_area)
    analysis = analyze_fingers(detection.contour if detection is not None else None)
    return FrameResult(mask, detection, analysis)
