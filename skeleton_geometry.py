
"""
Estimation geometrique du squelette de la main.

Les bases des doigts sont estimees a partir
des extremites et des creux entre les doigts.

Les articulations sont obtenues par interpolation.
La methode suppose une main approximativement
verticale, avec les doigts vers le haut.

Aucun modele d'apprentissage automatique.
"""

import numpy as np


def interpolate_point(start, end, ratio):
    """Calcule un point entre deux positions."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    point = start + ratio * (end - start)

    return tuple(np.rint(point).astype(int))


def estimate_palm_base(analysis):
    """
    Estime un point a la base de la paume,
    sans utiliser l'extremite de l'avant-bras.
    """
    center = analysis.get("center")
    fingertips = analysis.get("fingertips", [])

    if center is None:
        return None

    center_x, center_y = map(int, center)

    if analysis.get('palm_radius', 0) > 0 and analysis.get('wrist') is not None:
        return tuple(map(int, analysis['wrist']))

    if not fingertips:
        return center_x, center_y

    top_y = min(int(p[1]) for p in fingertips)
    height = max(0, center_y - top_y)

    return (
        center_x,
        center_y + round(0.45 * height),
    )


def find_valleys_between_fingers(tips, valleys, wrist):
    """
    Associe un creux a chaque espace entre
    deux extremites voisines.

    Retourne une liste de creux ou None.
    """
    gaps = []

    for i in range(len(tips) - 1):
        left_tip = tips[i]
        right_tip = tips[i + 1]

        left_x = min(left_tip[0], right_tip[0])
        right_x = max(left_tip[0], right_tip[0])

        candidates = [
            valley
            for valley in valleys
            if left_x < valley[0] < right_x
            and valley[1] > min(
                left_tip[1], right_tip[1]
            )
            and valley[1] < wrist[1]
        ]

        if not candidates:
            gaps.append(None)
            continue

        middle_x = (left_x + right_x) / 2

        chosen = min(
            candidates,
            key=lambda p: abs(p[0] - middle_x),
        )

        gaps.append(chosen)

    return gaps


def estimate_finger_bases(tips, valleys, wrist):
    """
    Estime les articulations de base des doigts.

    Utilise la hauteur des creux voisins pour
    placer chaque base sur la direction allant
    du poignet vers l'extremite du doigt.
    """
    gaps = find_valleys_between_fingers(
        tips, valleys, wrist
    )

    bases = []

    for i, tip in enumerate(tips):
        adjacent = []

        if i > 0 and gaps[i - 1] is not None:
            adjacent.append(gaps[i - 1])

        if i < len(gaps) and gaps[i] is not None:
            adjacent.append(gaps[i])

        if adjacent:
            valley_y = float(np.mean([
                p[1] for p in adjacent
            ]))

            vertical_distance = wrist[1] - tip[1]

            if vertical_distance > 1:
                ratio = (
                    (wrist[1] - valley_y)
                    / vertical_distance
                )
            else:
                ratio = 0.55

            # Garder les bases dans une zone
            # plausible de la main.
            ratio = float(
                np.clip(ratio, 0.43, 0.68)
            )
        else:
            # Estimation par defaut lorsqu'aucun
            # creux voisin n'est disponible.
            ratio = 0.55

        base = interpolate_point(
            wrist, tip, ratio
        )

        bases.append(base)

    return bases


def estimate_skeleton(analysis):
    """
    Construit un squelette geometrique simplifie.

    Avec cinq extremites :
    - 1 point de reference au poignet
    - 4 points par doigt

    Total : 21 points estimes.
    """
    result = {
        "detected": False,
        "points": [],
        "connections": [],
        "finger_chains": [],
    }

    if not analysis.get("detected", False):
        return result

    wrist = estimate_palm_base(analysis)

    if wrist is None:
        return result

    fingertips = sorted(
        [
            tuple(map(int, p))
            for p in analysis.get("fingertips", [])
        ],
        key=lambda p: p[0],
    )[:5]

    valleys = [
        tuple(map(int, p))
        for p in analysis.get("valleys", [])
    ]

    bases = estimate_finger_bases(
        fingertips,
        valleys,
        wrist,
    )

    points = [wrist]
    connections = []
    finger_chains = []

    for tip, base in zip(fingertips, bases):
        # Articulations intermediaires situees
        # entre la base et l'extremite du doigt.
        joint_1 = interpolate_point(
            base, tip, 1 / 3
        )

        joint_2 = interpolate_point(
            base, tip, 2 / 3
        )

        chain_points = [
            base,
            joint_1,
            joint_2,
            tip,
        ]

        start = len(points)
        points.extend(chain_points)

        chain = list(range(start, start + 4))
        finger_chains.append(chain)

        # Connexion dans la paume.
        connections.append((0, chain[0]))

        # Connexions le long du doigt.
        for j in range(3):
            connections.append(
                (chain[j], chain[j + 1])
            )

    result.update({
        "detected": True,
        "points": points,
        "connections": connections,
        "finger_chains": finger_chains,
    })

    return result
