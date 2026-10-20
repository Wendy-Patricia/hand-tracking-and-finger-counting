
"""
Analyse geometrique des doigts a partir d'un contour OpenCV.

Aucune acquisition video ni segmentation de couleur
dans ce module.

Hypothese principale :
main approximativement verticale, doigts vers le haut.
"""

import math
import cv2 as cv
import numpy as np


def _empty():
    """Retourne une analyse vide."""

    return dict(
        detected=False,
        count=0,
        fingertips=[],
        valleys=[],
        thumb=None,
        hull=None,
        center=None,
        wrist=None,
        skeleton=[],
        reliable=False
    )


def _fingertip_candidates(contour, mask, center, radius):
    """
    Recherche les extremites des doigts par
    analyse du profil radial et de la largeur locale.
    """

    pts = contour[:, 0, :].astype(np.float64)

    distances = np.linalg.norm(
        pts - center, axis=1
    )

    n = len(distances)

    if n < 15:
        return []

    # 1. Lissage du profil radial
    kernel_size = max(
        7, int(n * 0.018) | 1
    )

    kernel = (
        np.ones(kernel_size) / kernel_size
    )

    padded = np.r_[
        distances[-kernel_size:],
        distances,
        distances[:kernel_size]
    ]

    smooth = np.convolve(
        padded, kernel, mode="same"
    )[kernel_size:-kernel_size]

    # 2. Carte des distances interieures
    dist_inside = cv.distanceTransform(
        mask, cv.DIST_L2, 5
    )

    peaks = []

    search_window = min(
        kernel_size * 4,
        max(2, n // 4)
    )

    # 3. Detection des maxima locaux
    for i in range(n):

        previous = smooth[(i - 1) % n]
        following = smooth[(i + 1) % n]

        is_peak = (
            smooth[i] >= previous
            and smooth[i] > following
        )

        if not is_peak:
            continue

        # Distance minimale au centre de la paume
        if smooth[i] <= radius * 1.45:
            continue

        # Les doigts doivent se situer au-dessus
        # du centre de la paume
        if pts[i, 1] >= center[1] - 0.35 * radius:
            continue

        # 4. Verification de la proeminence
        local_min = min(
            smooth[(i - j) % n]
            for j in range(1, search_window)
        )

        prominence = smooth[i] - local_min

        if prominence <= 0.08 * radius:
            continue

        # 5. Verification de la largeur du doigt
        tip = pts[i]

        direction = center - tip
        norm = np.linalg.norm(direction)

        if norm == 0:
            continue

        # Avancer vers l'interieur du doigt
        inner = (
            tip
            + (direction / norm) * (0.45 * radius)
        )

        sx, sy = np.rint(inner).astype(int)

        if not (
            0 <= sx < mask.shape[1]
            and 0 <= sy < mask.shape[0]
        ):
            continue

        local_half_width = dist_inside[sy, sx]

        # Rejeter les regions trop larges,
        # notamment les coins d'un poing ferme
        if (
            local_half_width > 0.28 * radius
            or local_half_width < 0.035 * radius
        ):
            continue

        point = tuple(
            map(int, np.rint(tip))
        )

        peaks.append(
            (float(smooth[i]), point)
        )

    # 6. Elimination des detections multiples
    peaks.sort(reverse=True)

    selected = []

    for _, tip in peaks:

        if all(
            math.dist(tip, other) > 0.38 * radius
            for other in selected
        ):
            selected.append(tip)

    # Limiter a cinq extremites
    return sorted(
        selected[:5],
        key=lambda p: p[0]
    )


def analyze_fingers(contour):
    """
    Analyse une main a partir d'un contour OpenCV.

    Parametre :
        contour : tableau OpenCV (N, 1, 2) ou None

    Retour :
        dictionnaire contenant :
        - detected
        - count
        - fingertips
        - valleys
        - thumb
        - hull
        - center
        - wrist
        - skeleton
        - reliable
    """

    result = _empty()

    # 1. Verification du contour
    if contour is None:
        return result

    contour = np.asarray(
        contour, dtype=np.int32
    ).reshape(-1, 1, 2)

    if (
        len(contour) < 15
        or cv.contourArea(contour) < 1000
    ):
        return result

    x, y, w, h = cv.boundingRect(contour)

    # 2. Construction d'un masque interne
    # uniquement a partir du contour recu
    mask = np.zeros(
        (y + h + 2, x + w + 2),
        dtype=np.uint8
    )

    cv.drawContours(
        mask, [contour], -1, 255, -1
    )

    # Retablir un contour dense, meme si le
    # module A utilise CHAIN_APPROX_SIMPLE
    dense_contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )

    if not dense_contours:
        return result

    contour = max(
        dense_contours,
        key=cv.contourArea
    )

    # 3. Estimation du centre de la paume
    radius_map = cv.distanceTransform(
        mask,
        cv.DIST_L2,
        5
    )

    y_start = int(y + 0.38 * h)
    y_end = int(y + 0.78 * h)

    candidate_map = radius_map.copy()

    candidate_map[:y_start] = 0
    candidate_map[y_end:] = 0

    _, radius, _, palm_point = cv.minMaxLoc(
        candidate_map
    )

    if radius < 5:
        return result

    center = np.array(
        palm_point, dtype=np.float64
    )

    center_tuple = tuple(
        map(int, center)
    )

    # 4. Recherche des extremites des doigts
    fingertips = _fingertip_candidates(
        contour,
        mask,
        center,
        radius
    )

    # 5. Enveloppe convexe
    hull = cv.convexHull(contour)

    # 6. Defauts de convexite
    valleys = []

    hull_indices = cv.convexHull(
        contour,
        returnPoints=False
    )

    if (
        hull_indices is not None
        and len(hull_indices) >= 4
    ):
        try:
            defects = cv.convexityDefects(
                contour,
                hull_indices
            )
        except cv.error:
            defects = None

        if defects is not None:

            for s, e, f, depth in defects.reshape(-1, 4):

                far = tuple(
                    map(int, contour[int(f), 0])
                )

                # Profondeur du creux
                if depth / 256.0 < 0.20 * radius:
                    continue

                # Exclure les creux trop bas
                if far[1] >= center[1] + 0.45 * radius:
                    continue

                # Eliminer les doublons
                if all(
                    math.dist(far, q) > 0.30 * radius
                    for q in valleys
                ):
                    valleys.append(far)

    # 7. Estimation du poignet
    lower = contour[:, 0, :]

    bottom = lower[
        lower[:, 1] >= y + 0.92 * h
    ]

    if len(bottom):

        wrist = (
            int(np.median(bottom[:, 0])),
            int(np.median(bottom[:, 1]))
        )

    else:
        wrist = (
            int(center[0]),
            y + h - 1
        )

    # 8. Squelette geometrique simplifie
    skeleton = [
        (wrist, center_tuple)
    ]

    skeleton += [
        (center_tuple, tip)
        for tip in fingertips
    ]

    # 9. Comptage final
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
        reliable=(1 <= count <= 5)
    )

    return result


def count_fingers(contour):
    """Interface compatible avec les anciens appels."""

    data = analyze_fingers(contour)

    return (
        data["count"],
        data["fingertips"],
        data["valleys"],
        data["thumb"]
    )
