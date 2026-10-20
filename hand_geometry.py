
import cv2 as cv
import numpy as np


def extract_hand_contour(image):
    """Extrai o maior contorno externo da imagem."""

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv.threshold(
        blurred, 0, 255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    contour = max(contours, key=cv.contourArea)

    if cv.contourArea(contour) < 1000:
        return None, mask

    return contour, mask


def analyze_hand_geometry(contour):
    """Calcula o Convex Hull e os Convexity Defects."""

    # Convex Hull com coordenadas
    hull_points = cv.convexHull(contour)

    # Convex Hull com indices para os defects
    hull_indices = cv.convexHull(
        contour, returnPoints=False
    )

    defects = None

    if hull_indices is not None and len(hull_indices) > 3:
        defects = cv.convexityDefects(
            contour, hull_indices
        )

    return hull_points, defects
