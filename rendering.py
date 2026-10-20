"""Annotations visuelles des résultats de détection."""

import cv2 as cv
import numpy as np

from hand_detection import HandDetection


def draw_detection(frame: np.ndarray, detection: HandDetection | None) -> np.ndarray:
    """Dessine sur l'image fournie ; la copie éventuelle appartient à l'appelant."""
    if detection is None:
        cv.putText(frame, "Aucune main candidate", (15, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        return frame
    cv.drawContours(frame, [detection.contour], -1, (0, 255, 0), 2)
    cv.drawContours(frame, [detection.hull], -1, (255, 0, 0), 2)
    x, y, width, height = detection.bounding_box
    cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
    cv.putText(frame, detection.shape, (15, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv.putText(frame, f"Aire : {detection.area:.0f} px | Solidite : {detection.solidity:.2f}",
               (15, 100), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame
