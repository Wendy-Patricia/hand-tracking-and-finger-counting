"""Traitement d'une image, utilisable sans caméra ni fenêtre OpenCV."""

from dataclasses import dataclass, field

import numpy as np

from config import ProcessingParameters
from hand_detection import HandDetection, contour_from_mask, describe_hand
from finger_counting import count_fingers
from segmentation import segment_hand


@dataclass
class FrameResult:
    mask: np.ndarray
    detection: HandDetection | None
    count: int = 0
    fingertips: list[tuple[int, int]] = field(default_factory=list)

    @property
    def contour(self) -> np.ndarray | None:
        return self.detection.contour if self.detection is not None else None


def process_frame(frame: np.ndarray, parameters: ProcessingParameters) -> FrameResult:
    """Segmente puis analyse une image BGR sans modifier l'image source."""
    _, mask = segment_hand(frame, parameters)
    contour = contour_from_mask(mask, parameters.min_area)
    count, fingertips = count_fingers(contour)
    detection = describe_hand(contour) if contour is not None else None
    return FrameResult(mask, detection, count, fingertips)
