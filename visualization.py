
"""
Visualisation du squelette geometrique de la main.

Module B - Applications graphiques.

Affichage :
- Contour de la main
- Enveloppe convexe
- Creux entre les doigts
- Squelette geometrique
- Articulations estimees
- Extremites des doigts
- Nombre de doigts
"""

import cv2 as cv

from skeleton_geometry import estimate_skeleton


# Couleurs au format BGR (OpenCV)
COLOR_CONTOUR = (0, 200, 0)
COLOR_HULL = (255, 100, 0)
COLOR_SKELETON = (180, 40, 0)
COLOR_JOINT = (0, 220, 255)
COLOR_TIP = (0, 0, 255)
COLOR_WRIST = (255, 0, 255)
COLOR_VALLEY = (0, 100, 255)


def draw_results(frame, contour, analysis):
    """
    Dessine les resultats sur une image couleur.

    Compatible avec une image statique
    ou une image extraite d'une video.
    """

    output = frame.copy()

    if contour is None or not analysis.get(
        "detected", False
    ):
        cv.putText(
            output,
            "Aucune main detectee",
            (15, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )
        return output

    # 1. Contour de la main
    cv.drawContours(
        output,
        [contour],
        -1,
        COLOR_CONTOUR,
        2
    )

    # 2. Enveloppe convexe
    hull = analysis.get("hull")

    if hull is not None:
        cv.drawContours(
            output,
            [hull],
            -1,
            COLOR_HULL,
            2
        )

    # 3. Creux entre les doigts
    for valley in analysis.get("valleys", []):
        cv.circle(
            output,
            tuple(map(int, valley)),
            6,
            COLOR_VALLEY,
            -1,
            cv.LINE_AA
        )

    # 4. Calculer le squelette
    skeleton = estimate_skeleton(analysis)

    points = skeleton["points"]
    connections = skeleton["connections"]
    finger_chains = skeleton["finger_chains"]

    # 5. Dessiner les connexions du squelette
    for start_id, end_id in connections:

        start = tuple(map(int, points[start_id]))
        end = tuple(map(int, points[end_id]))

        # Bordure claire pour rester visible
        # sur les fonds sombres
        cv.line(
            output,
            start,
            end,
            (255, 255, 255),
            5,
            cv.LINE_AA
        )

        # Ligne principale bleu fonce
        cv.line(
            output,
            start,
            end,
            COLOR_SKELETON,
            3,
            cv.LINE_AA
        )

    # 6. Identifier les extremites
    fingertip_ids = {
        chain[-1]
        for chain in finger_chains
    }

    # 7. Dessiner les articulations
    for index, point in enumerate(points):

        point = tuple(map(int, point))

        if index == 0:
            color = COLOR_WRIST
            radius = 7

        elif index in fingertip_ids:
            color = COLOR_TIP
            radius = 7

        else:
            color = COLOR_JOINT
            radius = 5

        # Bordure noire
        cv.circle(
            output,
            point,
            radius + 2,
            (0, 0, 0),
            -1,
            cv.LINE_AA
        )

        # Point colore
        cv.circle(
            output,
            point,
            radius,
            color,
            -1,
            cv.LINE_AA
        )

    # 8. Afficher le comptage
    count = analysis.get("count", 0)

    cv.putText(
        output,
        f"Doigts estimes : {count}",
        (15, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # 9. Afficher les creux
    cv.putText(
        output,
        f"Creux detectes : {len(analysis.get('valleys', []))}",
        (15, 67),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # 10. Afficher le nombre de points
    cv.putText(
        output,
        f"Squelette estime : {len(points)} points",
        (15, 97),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # 11. Avertissement
    if not analysis.get("reliable", False):
        cv.putText(
            output,
            "Comptage incertain",
            (15, 127),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )

    return output
