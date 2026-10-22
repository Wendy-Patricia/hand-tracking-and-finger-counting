"""Analyse géométrique du plus grand contour candidat à une main."""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from config import ProcessingParameters
from segmentation import segment_hand


@dataclass
class HandDetection:
    contour: np.ndarray
    hull: np.ndarray
    bounding_box: tuple[int, int, int, int]
    area: float
    solidity: float
    shape: str


def detect_hand(
    frame: np.ndarray, parameters: ProcessingParameters | None = None
) -> np.ndarray | None:
    """Pessoa A: recebe um frame BGR e devolve o contorno da mão ou None.

    O argumento opcional permite usar os ajustes HSV da interface. O frame
    original não é alterado.
    """
    parameters = parameters if parameters is not None else ProcessingParameters()
    _, mask = segment_hand(frame, parameters)
    return contour_from_mask(mask, parameters.min_area)


def contour_from_mask(mask: np.ndarray, min_area: int = 3000) -> np.ndarray | None:
    """Retient le plus grand contour externe dépassant l'aire minimale.

    La description de forme est une heuristique géométrique, pas une
    reconnaissance fiable de gestes ni une preuve de présence d'une main.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(contour)
    if area <= 0 or area < min_area:
        return None
    return contour


def describe_hand(contour: np.ndarray) -> HandDetection:
    """Calcule les annotations sans changer le contrat public de détection."""
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = area / hull_area if hull_area else 0.0
    box = cv.boundingRect(contour)
    # Le rectangle orienté permet une description indépendante de la rotation.
    _, (width, height), _ = cv.minAreaRect(contour)
    elongation = max(width, height) / max(min(width, height), 1.0)
    if solidity < 0.80:
        shape = "Silhouette decoupee (doigts ecartes possibles)"
    elif elongation > 1.8:
        shape = "Silhouette allongee"
    else:
        shape = "Silhouette compacte"
    return HandDetection(contour, hull, box, area, solidity, shape)
