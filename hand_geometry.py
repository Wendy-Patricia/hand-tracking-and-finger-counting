"""Comptage géométrique des doigts, sans modèle d'apprentissage.

Le poignet doit pointer vers le bas. Les seuils de longueur sont exprimés
par rapport au rayon de la paume pour s'adapter à la taille de la main.
"""

import math

import cv2 as cv
import numpy as np


MAX_FINGERS = 5
MIN_CONTOUR_AREA = 1000


def _empty_analysis():
    return {
        "detected": False,
        "count": 0,
        "fingertips": [],
        "valleys": [],
        "thumb": None,
        "center": None,
        "wrist": None,
        "palm_radius": 0.0,
        "reliable": False,
    }


def _angle_between(first, second):
    """Renvoie l'angle entre deux vecteurs, en degrés."""
    denominator = max(np.linalg.norm(first) * np.linalg.norm(second), 1e-6)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1, 1))
    return math.degrees(math.acos(cosine))


def _resample_contour(contour, spacing):
    """Répartit les points à distance régulière le long du contour fermé."""
    points = contour[:, 0, :].astype(float)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    sample_count = max(32, int(cumulative[-1] / spacing))
    samples = np.linspace(0, cumulative[-1], sample_count, endpoint=False)
    return np.column_stack([
        np.interp(samples, cumulative, closed[:, axis]) for axis in range(2)
    ])


def _find_fingertips(contour, center, radius):
    points = _resample_contour(contour, max(1.0, radius * 0.025))
    # Le lissage circulaire évite de dépendre du point de départ du contour.
    points = sum(np.roll(points, shift, axis=0) for shift in range(-2, 3)) / 5
    distances = np.linalg.norm(points - center, axis=1)
    spacing = cv.arcLength(contour, True) / len(points)
    curvature_step = max(2, round(0.65 * radius / spacing))
    prominence_window = max(curvature_step, round(1.2 * radius / spacing))
    candidates = []

    for index, tip in enumerate(points):
        distance = distances[index]
        if distance < 1.4 * radius or tip[1] > center[1] + 0.6 * radius:
            continue

        previous = distances[index - 1]
        following = distances[(index + 1) % len(points)]
        if distance < previous or distance <= following:
            continue

        # Un pic doit descendre des deux côtés pour éliminer les petites bosses.
        before = min(
            distances[(index - step) % len(points)]
            for step in range(1, prominence_window + 1)
        )
        after = min(
            distances[(index + step) % len(points)]
            for step in range(1, prominence_window + 1)
        )
        prominence = distance - max(before, after)
        if prominence < 0.16 * radius:
            continue

        left = points[(index - curvature_step) % len(points)] - tip
        right = points[(index + curvature_step) % len(points)] - tip
        if _angle_between(left, right) > 85:
            continue

        point = tuple(map(int, np.rint(tip)))
        candidates.append((prominence, distance, point))

    # Garder le meilleur pic lorsque plusieurs candidats décrivent le même doigt.
    selected = []
    for _, _, tip in sorted(candidates, reverse=True):
        if all(math.dist(tip, other) > 0.55 * radius for other in selected):
            selected.append(tip)
    return sorted(selected, key=lambda point: point[0])


def _find_valleys(contour, center, radius):
    """Filtre les défauts de convexité pour retenir les creux entre les doigts."""
    hull_indices = cv.convexHull(contour, returnPoints=False)
    if len(hull_indices) < 4:
        return []
    try:
        defects = cv.convexityDefects(contour, hull_indices)
    except cv.error:
        return []
    if defects is None:
        return []

    valleys = []
    for start, end, far, depth in defects.reshape(-1, 4):
        left, right, valley = contour[[start, end, far], 0].astype(float)
        if depth / 256 < 0.3 * radius:
            continue
        if valley[1] > center[1] + 0.5 * radius:
            continue
        if _angle_between(left - valley, right - valley) > 100:
            continue
        if all(math.dist(valley, other) > 0.3 * radius for other in valleys):
            valleys.append(tuple(map(int, valley)))
    return valleys


def analyze_fingers(contour):
    """Analyse un contour et renvoie les points dans le repère de l'image.

    ``reliable`` indique une plausibilité géométrique. Une silhouette compacte
    sans extrémité peut valoir zéro ; ce critère ne prouve pas la présence d'une main.
    """
    result = _empty_analysis()
    if contour is None:
        return result

    contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    if len(contour) < 3 or cv.contourArea(contour) < MIN_CONTOUR_AREA:
        return result

    # Travailler dans un masque local évite une allocation liée à la position
    # de la main et accepte aussi les coordonnées négatives.
    x, y, width, height = cv.boundingRect(contour)
    offset = np.array([x - 2, y - 2])
    mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv.drawContours(mask, [(contour - offset).astype(np.int32)], -1, 255, -1)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    local_contour = max(contours, key=cv.contourArea).reshape(-1, 1, 2)

    # La largeur limite la recherche de paume pour exclure un avant-bras long.
    distance_map = cv.distanceTransform(mask, cv.DIST_L2, 5)
    palm_limit = min(height + 2, round(1.65 * width) + 2)
    _, radius, _, palm = cv.minMaxLoc(distance_map[:palm_limit])
    if radius < 5:
        return result

    center = np.asarray(palm, dtype=float)
    tips = _find_fingertips(local_contour, center, radius)
    valleys = _find_valleys(local_contour, center, radius)
    hull = cv.convexHull(local_contour)
    solidity = cv.contourArea(local_contour) / max(cv.contourArea(hull), 1)

    def to_image_point(point):
        return tuple(map(int, np.rint(np.asarray(point) + offset)))

    fingertips = [to_image_point(tip) for tip in tips[:MAX_FINGERS]]
    palm_center = to_image_point(center)
    # N'attribuer le pouce que lorsqu'un seul candidat est clairement latéral.
    lateral_tips = [
        tip for tip in fingertips
        if abs(tip[0] - palm_center[0]) > 1.25 * radius
        and tip[1] > palm_center[1] - 0.65 * radius
    ]
    result.update(
        detected=True,
        count=len(fingertips),
        fingertips=fingertips,
        valleys=[to_image_point(valley) for valley in valleys],
        thumb=lateral_tips[0] if len(lateral_tips) == 1 else None,
        center=palm_center,
        wrist=to_image_point(center + [0, radius]),
        palm_radius=float(radius),
        reliable=len(tips) <= MAX_FINGERS and (bool(tips) or solidity >= 0.85),
    )
    return result
