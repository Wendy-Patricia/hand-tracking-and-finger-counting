"""Adaptation des curseurs OpenCV vers les paramètres du traitement."""

import cv2 as cv

from config import ProcessingParameters


WINDOW = "Reglages HSV"
# Attribut métier, libellé affiché, valeur maximale du curseur.
TRACKBARS = (
    ("h_min", "H min", 179),
    ("h_max", "H max", 179),
    ("s_min", "S min", 255),
    ("s_max", "S max", 255),
    ("v_min", "V min", 255),
    ("v_max", "V max", 255),
    ("blur", "Flou", 15),
    ("erosions", "Erosions", 5),
    ("dilations", "Dilatations", 5),
    ("min_area", "Aire min", 30000),
)


def create_trackbars():
    """Crée les curseurs avec les valeurs par défaut du traitement."""
    cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW, 470, 420)
    defaults = ProcessingParameters()
    for field, label, limit in TRACKBARS:
        cv.createTrackbar(label, WINDOW, getattr(defaults, field), limit, lambda _: None)


def read_parameters() -> ProcessingParameters:
    return ProcessingParameters(**{
        field: cv.getTrackbarPos(label, WINDOW)
        for field, label, _ in TRACKBARS
    })


def reset_trackbars():
    defaults = ProcessingParameters()
    for field, label, _ in TRACKBARS:
        cv.setTrackbarPos(label, WINDOW, getattr(defaults, field))
