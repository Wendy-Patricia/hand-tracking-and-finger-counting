"""Segmentation de la main par couleur dans l'espace HSV."""

import cv2 as cv
import numpy as np

from config import ProcessingParameters


def segment_hand(
    frame: np.ndarray, parameters: ProcessingParameters
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie une image HSV et un masque binaire nettoyé.

    Une teinte minimale supérieure à la maximale sélectionne un intervalle
    circulaire (par exemple 170..179 et 0..15 pour les rouges).
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h_min, h_max = parameters.h_min, parameters.h_max
    s_min, s_max = sorted((parameters.s_min, parameters.s_max))
    v_min, v_max = sorted((parameters.v_min, parameters.v_max))
    if h_min <= h_max:
        mask = cv.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
    else:
        mask = cv.bitwise_or(
            cv.inRange(hsv, (h_min, s_min, v_min), (179, s_max, v_max)),
            cv.inRange(hsv, (0, s_min, v_min), (h_max, s_max, v_max)),
        )

    # Une taille de 1 signifie aucun lissage.
    size = 2 * parameters.blur + 1
    mask = cv.GaussianBlur(mask, (size, size), 0)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.erode(mask, kernel, iterations=parameters.erosions)
    mask = cv.dilate(mask, kernel, iterations=parameters.dilations)
    return hsv, mask
