
"""
Estimation geometrique du squelette de la main.

Module B - Applications graphiques.

Methodes utilisees :
- Geometrie des points
- Interpolation lineaire
- Distances euclidiennes
- Connexions entre points

Aucun modele d'apprentissage automatique.
"""

import numpy as np


def interpolate_point(start, end, ratio):
    """
    Calcule un point intermediaire entre
    deux coordonnees.

    ratio = 0 : point de depart
    ratio = 1 : point d'arrivee
    """

    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)

    point = a + ratio * (b - a)

    return tuple(
        np.rint(point).astype(int)
    )


def estimate_skeleton(analysis):
    """
    Construit un squelette geometrique simplifie.

    Parametre :
        analysis : resultat de analyze_fingers()

    Retour :
        {
            "points": liste des points,
            "connections": liste des connexions,
            "finger_chains": chaines des doigts,
            "detected": bool
        }

    Les articulations intermediaires sont
    estimees, et non detectees anatomiquement.
    """

    result = {
        "detected": False,
        "points": [],
        "connections": [],
        "finger_chains": []
    }

    if not analysis.get("detected", False):
        return result

    center = analysis.get("center")
    wrist = analysis.get("wrist")
    fingertips = analysis.get("fingertips", [])

    if center is None or wrist is None:
        return result

    # Reference du poignet : point numero 0
    points = [
        tuple(map(int, wrist))
    ]

    connections = []
    finger_chains = []

    # Ordonner les doigts de gauche a droite
    fingertips = sorted(
        fingertips,
        key=lambda p: p[0]
    )

    # Construire une chaine pour chaque doigt
    for tip in fingertips[:5]:

        tip = tuple(map(int, tip))

        # Base approximative du doigt
        base = interpolate_point(
            center, tip, 0.30
        )

        # Articulation intermediaire proximale
        joint_1 = interpolate_point(
            center, tip, 0.58
        )

        # Articulation intermediaire distale
        joint_2 = interpolate_point(
            center, tip, 0.79
        )

        # Extremite du doigt
        finger_points = [
            base,
            joint_1,
            joint_2,
            tip
        ]

        start_index = len(points)

        points.extend(finger_points)

        # Relier les articulations du doigt
        chain = list(
            range(start_index, start_index + 4)
        )

        finger_chains.append(chain)

        for i in range(3):
            connections.append(
                (chain[i], chain[i + 1])
            )

    # Relier les bases des doigts entre elles
    for i in range(len(finger_chains) - 1):
        connections.append(
            (
                finger_chains[i][0],
                finger_chains[i + 1][0]
            )
        )

    # Relier le poignet a la paume
    if finger_chains:
        middle = len(finger_chains) // 2

        connections.append(
            (0, finger_chains[middle][0])
        )

    result.update({
        "detected": True,
        "points": points,
        "connections": connections,
        "finger_chains": finger_chains
    })

    return result
