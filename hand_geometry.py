
"""
Analyse geometrique des doigts a partir d'un contour OpenCV.

Le contour est fourni par le module de detection.
La methode suppose une main approximativement verticale,
avec les doigts orientes vers le haut.
"""

import math

import cv2 as cv
import numpy as np


def _empty():
    """Retourne une analyse vide."""
    return {
        "detected": False,
        "count": 0,
        "fingertips": [],
        "valleys": [],
        "thumb": None,
        "hull": None,
        "center": None,
        "wrist": None,
        "skeleton": [],
        "reliable": False,
    }


def _fingertip_candidates(contour, mask, center, radius):
    """
    Recherche les extremites des doigts a partir
    du profil radial et de la largeur locale.
    """
    points = contour[:, 0, :].astype(np.float64)
    distances = np.linalg.norm(points - center, axis=1)
    n = len(distances)

    if n < 15:
        return []

    # Lisser le profil radial.
    kernel_size = max(7, int(n * 0.018) | 1)
    kernel = np.ones(kernel_size) / kernel_size

    padded = np.r_[
        distances[-kernel_size:],
        distances,
        distances[:kernel_size],
    ]

    smooth = np.convolve(
        padded,
        kernel,
        mode="same",
    )[kernel_size:-kernel_size]

    distance_map = cv.distanceTransform(
        mask,
        cv.DIST_L2,
        5,
    )

    search_window = min(
        kernel_size * 4,
        max(2, n // 4),
    )

    candidates = []

    for i in range(n):
        previous = smooth[(i - 1) % n]
        following = smooth[(i + 1) % n]

        # Maximum local du profil radial.
        if not (
            smooth[i] >= previous
            and smooth[i] > following
        ):
            continue

        # Distance minimale au centre de la paume.
        if smooth[i] <= radius * 1.45:
            continue

        # Les doigts sont supposes au-dessus de la paume.
        if points[i, 1] >= center[1] - 0.35 * radius:
            continue

        # Verifier la prominence du maximum.
        local_min = min(
            smooth[(i - j) % n]
            for j in range(1, search_window)
        )

        if smooth[i] - local_min <= 0.08 * radius:
            continue

        # Verifier la largeur locale du doigt.
        tip = points[i]
        direction = center - tip
        norm = np.linalg.norm(direction)

        if norm == 0:
            continue

        inner = tip + (
            direction / norm
        ) * (0.45 * radius)

        sx, sy = np.rint(inner).astype(int)

        if not (
            0 <= sx < mask.shape[1]
            and 0 <= sy < mask.shape[0]
        ):
            continue

        local_half_width = distance_map[sy, sx]

        if (
            local_half_width > 0.28 * radius
            or local_half_width < 0.035 * radius
        ):
            continue

        fingertip = tuple(
            map(int, np.rint(tip))
        )

        candidates.append(
            (float(smooth[i]), fingertip)
        )

    # Eviter plusieurs detections sur le meme doigt.
    candidates.sort(reverse=True)
    selected = []

    for _, tip in candidates:
        if all(
            math.dist(tip, other) > 0.38 * radius
            for other in selected
        ):
            selected.append(tip)

    return sorted(
        selected[:5],
        key=lambda point: point[0],
    )


def analyze_fingers(contour):
    """
    Analyse les doigts a partir d'un contour OpenCV.

    Retourne un dictionnaire contenant le comptage,
    les extremites, les creux, la paume et le poignet.
    """
    result = _empty()

    if contour is None:
        return result

    contour = np.asarray(
        contour,
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    if (
        len(contour) < 15
        or cv.contourArea(contour) < 1000
    ):
        return result

    x, y, width, height = cv.boundingRect(contour)

    # Reconstruire un contour dense a partir du contour
    # fourni par le module de detection.
    mask = np.zeros(
        (y + height + 2, x + width + 2),
        dtype=np.uint8,
    )

    cv.drawContours(
        mask,
        [contour],
        -1,
        255,
        -1,
    )

    dense_contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE,
    )

    if not dense_contours:
        return result

    contour = max(
        dense_contours,
        key=cv.contourArea,
    )

    # Estimer le centre de la paume.
    radius_map = cv.distanceTransform(
        mask,
        cv.DIST_L2,
        5,
    )

    y_start = int(y + 0.38 * height)
    y_end = int(y + 0.78 * height)

    candidate_map = radius_map.copy()
    candidate_map[:y_start] = 0
    candidate_map[y_end:] = 0

    _, radius, _, palm_point = cv.minMaxLoc(
        candidate_map
    )

    if radius < 5:
        return result

    center = np.array(
        palm_point,
        dtype=np.float64,
    )
    center_tuple = tuple(map(int, center))

    # Detecter les extremites des doigts.
    fingertips = _fingertip_candidates(
        contour,
        mask,
        center,
        radius,
    )

    # Calculer l'enveloppe convexe.
    hull = cv.convexHull(contour)
    hull_indices = cv.convexHull(
        contour,
        returnPoints=False,
    )

    # Detecter les creux entre les doigts.
    valleys = []

    if (
        hull_indices is not None
        and len(hull_indices) >= 4
    ):
        try:
            defects = cv.convexityDefects(
                contour,
                hull_indices,
            )
        except cv.error:
            defects = None

        if defects is not None:
            for _, _, far_index, depth in defects.reshape(
                -1, 4
            ):
                valley = tuple(
                    map(
                        int,
                        contour[int(far_index), 0],
                    )
                )

                if depth / 256.0 < 0.20 * radius:
                    continue

                if valley[1] >= center[1] + 0.45 * radius:
                    continue

                if all(
                    math.dist(valley, other) > 0.30 * radius
                    for other in valleys
                ):
                    valleys.append(valley)

    # Estimer la position du poignet.
    contour_points = contour[:, 0, :]

    bottom = contour_points[
        contour_points[:, 1] >= y + 0.92 * height
    ]

    if len(bottom):
        wrist = (
            int(np.median(bottom[:, 0])),
            int(np.median(bottom[:, 1])),
        )
    else:
        wrist = (
            int(center[0]),
            y + height - 1,
        )

    # Representation simple conservee pour compatibilite.
    # Le squelette affiche est calcule dans skeleton_geometry.py.
    skeleton = [(wrist, center_tuple)]
    skeleton += [
        (center_tuple, tip)
        for tip in fingertips
    ]

    count = len(fingertips)

    result.update(
        detected=True,
        count=count,
        fingertips=fingertips,
        valleys=valleys,
        thumb=None,
        hull=hull,
        center=center_tuple,
        wrist=wrist,
        skeleton=skeleton,
        reliable=(1 <= count <= 5),
    )

    return result


def count_fingers(contour):
    """Conserve l'interface utilisee par les anciens tests."""
    analysis = analyze_fingers(contour)

    return (
        analysis["count"],
        analysis["fingertips"],
        analysis["valleys"],
        analysis["thumb"],
    )
