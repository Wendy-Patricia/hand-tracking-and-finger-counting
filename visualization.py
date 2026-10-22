"""Annotations de la silhouette, des doigts et du squelette sur une copie de l'image."""

import cv2 as cv

from skeleton_geometry import estimate_skeleton


WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 100, 255)
TIP_COLOR = (0, 0, 255)
PALM_COLOR = (255, 0, 255)
JOINT_COLOR = (0, 220, 255)
SKELETON_COLOR = (180, 40, 0)


def _draw_text(frame, text, y, color=WHITE, scale=0.6):
    cv.putText(frame, text, (15, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_hand_contour(frame, contour, analysis):
    """Dessine le contour en masquant la partie basse de l'avant-bras."""
    wrist = analysis["wrist"]
    tips = analysis["fingertips"]
    top_y = min((point[1] for point in tips), default=wrist[1])
    cutoff_y = int(wrist[1] + 0.15 * max(0, wrist[1] - top_y))
    height, width = frame.shape[:2]
    clip_rect = (0, 0, width, max(1, min(cutoff_y, height)))
    points = contour.reshape(-1, 2).tolist()

    for start, end in zip(points, points[1:] + points[:1]):
        visible, start, end = cv.clipLine(clip_rect, tuple(start), tuple(end))
        if visible:
            cv.line(frame, start, end, GREEN, 2, cv.LINE_AA)


def _draw_skeleton(frame, skeleton):
    points = skeleton["points"]
    for start_id, end_id in skeleton["connections"]:
        start, end = points[start_id], points[end_id]
        cv.line(frame, start, end, WHITE, 5, cv.LINE_AA)
        cv.line(frame, start, end, SKELETON_COLOR, 3, cv.LINE_AA)

    tip_ids = {chain[-1] for chain in skeleton["finger_chains"]}
    for index, point in enumerate(points):
        if index == 0:
            color, radius = PALM_COLOR, 7
        elif index in tip_ids:
            color, radius = TIP_COLOR, 7
        else:
            color, radius = JOINT_COLOR, 5
        cv.circle(frame, point, radius + 2, (0, 0, 0), -1, cv.LINE_AA)
        cv.circle(frame, point, radius, color, -1, cv.LINE_AA)


def draw_results(frame, contour, analysis):
    """Affiche la mesure, le comptage stabilisé et le squelette estimé."""
    output = frame.copy()
    if contour is None or not analysis["detected"]:
        _draw_text(output, "Aucune main detectee", 35, TIP_COLOR)
        return output

    draw_hand_contour(output, contour, analysis)
    for valley in analysis["valleys"]:
        cv.circle(output, valley, 5, ORANGE, -1, cv.LINE_AA)

    skeleton = estimate_skeleton(analysis)
    _draw_skeleton(output, skeleton)

    if "stable_count" in analysis:
        stable = analysis["stable_count"]
        label = f"Doigts stabilises : {stable if stable is not None else 'en attente'}"
    else:
        label = f"Doigts estimes : {analysis['count']}"

    _draw_text(output, label, 35, GREEN, scale=0.8)
    _draw_text(output, f"Mesure : {analysis['count']} | Creux : {len(analysis['valleys'])}", 67)
    _draw_text(output, f"Squelette estime : {len(skeleton['points'])} points", 97)
    if not analysis["reliable"]:
        _draw_text(output, "Comptage incertain", 127, ORANGE)
    return output


def draw_detection(frame, detection):
    """Affiche le contour candidat lorsque l'analyse des doigts est indisponible."""
    output = frame.copy()
    if detection is None:
        _draw_text(output, "Aucune main candidate", 70, ORANGE)
        return output

    cv.drawContours(output, [detection.contour], -1, GREEN, 2)
    cv.drawContours(output, [detection.hull], -1, (255, 0, 0), 2)
    x, y, width, height = detection.bounding_box
    cv.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    _draw_text(output, detection.shape, 70)
    _draw_text(output, f"Aire : {detection.area:.0f} px | Solidite : {detection.solidity:.2f}", 100)
    return output
