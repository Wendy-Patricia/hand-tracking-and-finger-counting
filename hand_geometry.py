"""Finger counting from silhouettes using only NumPy and OpenCV.

The wrist must point approximately downwards. Length thresholds are relative
to the inscribed palm radius, rather than the image resolution.
"""

import math

import cv2 as cv
import numpy as np


def _empty():
    return dict(detected=False, count=0, fingertips=[], valleys=[], thumb=None,
                hull=None, center=None, wrist=None, skeleton=[], reliable=False,
                palm_radius=0.0)


def _resample(contour, spacing):
    """Uniform arc-length sampling, independent of contour approximation."""
    points = contour[:, 0, :].astype(float)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    samples = np.linspace(0, cumulative[-1],
                          max(32, int(cumulative[-1] / spacing)), endpoint=False)
    return np.column_stack([np.interp(samples, cumulative, closed[:, axis])
                            for axis in range(2)])


def _fingertip_candidates(contour, center, radius):
    points = _resample(contour, max(1.0, radius * 0.025))
    # Circular smoothing makes the contour's starting point irrelevant.
    points = sum(np.roll(points, shift, axis=0) for shift in range(-2, 3)) / 5
    distances = np.linalg.norm(points - center, axis=1)
    spacing = cv.arcLength(contour, True) / len(points)
    step = max(2, round(0.65 * radius / spacing))
    window = max(step, round(1.2 * radius / spacing))
    candidates = []
    for i, tip in enumerate(points):
        distance = distances[i]
        if distance < 1.4 * radius or tip[1] > center[1] + 0.6 * radius:
            continue
        if distance < distances[i - 1] or distance <= distances[(i + 1) % len(points)]:
            continue
        # Require a fall on BOTH sides to reject noise and wrist corners.
        before = min(distances[(i - j) % len(points)] for j in range(1, window + 1))
        after = min(distances[(i + j) % len(points)] for j in range(1, window + 1))
        prominence = distance - max(before, after)
        if prominence < 0.16 * radius:
            continue
        left = points[(i - step) % len(points)] - tip
        right = points[(i + step) % len(points)] - tip
        cosine = np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-6)
        angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
        if angle > 85:
            continue
        candidates.append((prominence, distance, tuple(np.rint(tip).astype(int))))

    selected = []
    for _, _, tip in sorted(candidates, reverse=True):
        if all(math.dist(tip, other) > 0.55 * radius for other in selected):
            selected.append(tip)
    return sorted(selected, key=lambda tip: tip[0])


def analyze_fingers(contour):
    """Return count, palm, tips and valleys in the input coordinate system.

    ``reliable`` indicates geometric plausibility, not recognition of a hand.
    A well segmented fist is a valid zero. Joined or occluded fingers cannot
    be recovered reliably from a silhouette alone.
    """
    result = _empty()
    if contour is None:
        return result
    contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    # CHAIN_APPROX_SIMPLE may represent a valid silhouette with four points.
    if len(contour) < 3 or cv.contourArea(contour) < 1000:
        return result
    x, y, width, height = cv.boundingRect(contour)
    offset = np.array([x - 2, y - 2])
    local = contour - offset
    mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv.drawContours(mask, [local.astype(np.int32)], -1, 255, -1)
    dense, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    local = max(dense, key=cv.contourArea).reshape(-1, 1, 2)

    distance_map = cv.distanceTransform(mask, cv.DIST_L2, 5)
    # Exclude the lower forearm using hand WIDTH, not full contour height.
    palm_region = distance_map[:min(height + 2, round(1.65 * width) + 2)]
    _, radius, _, palm = cv.minMaxLoc(palm_region)
    if radius < 5:
        return result
    center = np.asarray(palm, dtype=float)
    tips = _fingertip_candidates(local, center, radius)

    hull_indices = cv.convexHull(local, returnPoints=False)
    valleys = []
    try:
        defects = cv.convexityDefects(local, hull_indices) if len(hull_indices) >= 4 else None
    except cv.error:
        defects = None
    if defects is not None:
        for start, end, far, depth in defects.reshape(-1, 4):
            a, b, v = local[[start, end, far], 0].astype(float)
            va, vb = a - v, b - v
            cosine = np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-6)
            if (depth / 256 < 0.3 * radius or cosine < math.cos(math.radians(100))
                    or v[1] > center[1] + 0.5 * radius):
                continue
            if all(math.dist(v, other) > 0.3 * radius for other in valleys):
                valleys.append(tuple(map(int, v)))

    # Palm-base reference instead of the end of a potentially long arm.
    wrist = center + [0, radius]

    def absolute(point):
        return tuple(map(int, np.rint(np.asarray(point) + offset)))

    fingertips = [absolute(tip) for tip in tips[:5]]
    center_tuple, wrist_tuple = absolute(center), absolute(wrist)
    hull = cv.convexHull(local).reshape(-1, 1, 2)
    solidity = cv.contourArea(local) / max(cv.contourArea(hull), 1)
    # Only label a thumb when there is one clear lateral candidate.
    lateral = [tip for tip in fingertips
               if abs(tip[0] - center_tuple[0]) > 1.25 * radius
               and tip[1] > center_tuple[1] - 0.65 * radius]
    result.update(
        detected=True, count=len(fingertips), fingertips=fingertips,
        valleys=[absolute(v) for v in valleys],
        thumb=lateral[0] if len(lateral) == 1 else None,
        hull=(hull + offset).astype(np.int32), center=center_tuple, wrist=wrist_tuple,
        palm_radius=float(radius),
        skeleton=[(wrist_tuple, center_tuple)] + [(center_tuple, tip) for tip in fingertips],
        reliable=(len(tips) <= 5 and (bool(tips) or solidity >= 0.85)),
    )
    return result


def count_fingers(contour):
    """Compatibility interface for callers expecting the original tuple."""
    analysis = analyze_fingers(contour)
    return (analysis['count'], analysis['fingertips'], analysis['valleys'], analysis['thumb'])
