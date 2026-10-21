"""Squelette géométrique : bases estimées et articulations interpolées.

Ces points servent à la visualisation, pas à mesurer les articulations réelles.
La main est supposée orientée vers le haut.
"""

import numpy as np


def interpolate_point(start, end, ratio):
    """Place un point à la fraction donnée du segment."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return tuple(map(int, np.rint(start + ratio * (end - start))))


def find_valleys_between_fingers(tips, valleys, wrist):
    """Associe un creux à chaque paire de doigts voisins, ou None."""
    gaps = []
    for left, right in zip(tips, tips[1:]):
        middle_x = (left[0] + right[0]) / 2
        candidates = [
            valley for valley in valleys
            if left[0] < valley[0] < right[0]
            and min(left[1], right[1]) < valley[1] < wrist[1]
        ]
        nearest = min(candidates, key=lambda point: abs(point[0] - middle_x), default=None)
        gaps.append(nearest)
    return gaps


def estimate_finger_bases(tips, valleys, wrist):
    """Place les bases selon la hauteur des creux voisins."""
    gaps = find_valleys_between_fingers(tips, valleys, wrist)
    bases = []
    for index, tip in enumerate(tips):
        adjacent = [
            gaps[gap_index] for gap_index in (index - 1, index)
            if 0 <= gap_index < len(gaps) and gaps[gap_index] is not None
        ]
        ratio = 0.55
        if adjacent:
            valley_y = float(np.mean([point[1] for point in adjacent]))
            if wrist[1] - tip[1] > 1:
                ratio = (wrist[1] - valley_y) / (wrist[1] - tip[1])
            ratio = float(np.clip(ratio, 0.43, 0.68))
        bases.append(interpolate_point(wrist, tip, ratio))
    return bases


def estimate_skeleton(analysis):
    """Construit un point de poignet et quatre points par doigt visible."""
    result = {"points": [], "connections": [], "finger_chains": []}
    if not analysis["detected"]:
        return result

    wrist = analysis["wrist"]
    tips = sorted(analysis["fingertips"], key=lambda point: point[0])
    bases = estimate_finger_bases(tips, analysis["valleys"], wrist)
    result["points"].append(wrist)

    for tip, base in zip(tips, bases):
        start = len(result["points"])
        chain = list(range(start, start + 4))
        result["points"].extend([
            base,
            interpolate_point(base, tip, 1 / 3),
            interpolate_point(base, tip, 2 / 3),
            tip,
        ])
        result["finger_chains"].append(chain)
        result["connections"].append((0, start))
        result["connections"].extend(zip(chain, chain[1:]))

    return result
