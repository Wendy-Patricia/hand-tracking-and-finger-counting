
"""
Visualisation des doigts et du squelette geometrique.

Le contour affiche est limite a la zone de la main.
L'analyse utilise toujours le contour original complet.
"""

import cv2 as cv

from skeleton_geometry import (
    estimate_palm_base,
    estimate_skeleton,
)


# Couleurs BGR
COLOR_CONTOUR = (0, 200, 0)
COLOR_HULL = (255, 100, 0)
COLOR_SKELETON = (180, 40, 0)
COLOR_JOINT = (0, 220, 255)
COLOR_TIP = (0, 0, 255)
COLOR_PALM = (255, 0, 255)
COLOR_VALLEY = (0, 100, 255)

# Options de visualisation
SHOW_CONTOUR = True
SHOW_HULL = False
SHOW_VALLEYS = True


def draw_hand_contour(frame, contour, analysis):
    """
    Dessine uniquement la partie superieure
    du contour, correspondant approximativement
    a la main.

    Ne modifie pas le contour utilise pour l'analyse.
    """
    palm_base = estimate_palm_base(analysis)

    if palm_base is None:
        return

    fingertips = analysis.get("fingertips", [])

    if fingertips:
        top_y = min(int(p[1]) for p in fingertips)
        hand_height = max(0, palm_base[1] - top_y)

        # Limite legerement sous la base de la paume.
        cutoff_y = int(
            palm_base[1] + 0.15 * hand_height
        )
    else:
        cutoff_y = palm_base[1]

    image_height, image_width = frame.shape[:2]

    cutoff_y = max(
        1,
        min(cutoff_y, image_height),
    )

    # Rectangle dans lequel le contour est visible.
    clip_rect = (
        0,
        0,
        image_width,
        cutoff_y,
    )

    points = contour.reshape(-1, 2)

    if len(points) < 2:
        return

    for i in range(len(points)):
        start = tuple(
            map(int, points[i])
        )
        end = tuple(
            map(int, points[(i + 1) % len(points)])
        )

        visible, clipped_start, clipped_end = cv.clipLine(
            clip_rect,
            start,
            end,
        )

        if visible:
            cv.line(
                frame,
                clipped_start,
                clipped_end,
                COLOR_CONTOUR,
                2,
                cv.LINE_AA,
            )


def draw_results(frame, contour, analysis):
    """
    Affiche l'analyse de la main sur une image BGR.
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
            2,
        )
        return output

    # Contour limite a la region de la main.
    if SHOW_CONTOUR:
        draw_hand_contour(
            output,
            contour,
            analysis,
        )

    # Enveloppe convexe facultative.
    if SHOW_HULL:
        hull = analysis.get("hull")

        if hull is not None:
            cv.drawContours(
                output,
                [hull],
                -1,
                COLOR_HULL,
                2,
            )

    # Creux entre les doigts.
    if SHOW_VALLEYS:
        for valley in analysis.get("valleys", []):
            cv.circle(
                output,
                tuple(map(int, valley)),
                5,
                COLOR_VALLEY,
                -1,
                cv.LINE_AA,
            )

    # Squelette geometrique.
    skeleton = estimate_skeleton(analysis)

    points = skeleton["points"]
    connections = skeleton["connections"]
    finger_chains = skeleton["finger_chains"]

    # Connexions.
    for start_id, end_id in connections:
        start = tuple(map(int, points[start_id]))
        end = tuple(map(int, points[end_id]))

        cv.line(
            output,
            start,
            end,
            (255, 255, 255),
            5,
            cv.LINE_AA,
        )

        cv.line(
            output,
            start,
            end,
            COLOR_SKELETON,
            3,
            cv.LINE_AA,
        )

    # Extremites identifiees.
    fingertip_ids = {
        chain[-1]
        for chain in finger_chains
    }

    # Points du squelette.
    for index, point in enumerate(points):
        point = tuple(map(int, point))

        if index == 0:
            color = COLOR_PALM
            radius = 7
        elif index in fingertip_ids:
            color = COLOR_TIP
            radius = 7
        else:
            color = COLOR_JOINT
            radius = 5

        cv.circle(
            output,
            point,
            radius + 2,
            (0, 0, 0),
            -1,
            cv.LINE_AA,
        )

        cv.circle(
            output,
            point,
            radius,
            color,
            -1,
            cv.LINE_AA,
        )

    # Informations affichees.
    stable = analysis.get('stable_count')
    # Offline images have no temporal filter; keep their raw count visible.
    count_label = (f"Doigts stabilises : {stable}" if stable is not None
                   else "Doigts stabilises : en attente"
                   if 'stable_count' in analysis
                   else f"Doigts estimes : {analysis.get('count', 0)}")
    cv.putText(
        output,
        count_label,
        (15, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv.putText(
        output,
        f"Mesure : {analysis.get('count', 0)} | Creux : {len(analysis.get('valleys', []))}",
        (15, 67),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv.putText(
        output,
        f"Squelette estime : {len(points)} points",
        (15, 97),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if not analysis.get("reliable", False):
        cv.putText(
            output,
            "Comptage incertain",
            (15, 127),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )

    return output
