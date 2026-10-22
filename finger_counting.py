"""Pessoa B: contagem geométrica para uma mão com os dedos para cima."""

import math

import cv2 as cv
import numpy as np


def count_fingers(contour: np.ndarray | None) -> tuple[int, list[tuple[int, int]]]:
    """Devolve (count, fingertips), com pontas em coordenadas (x, y) do frame.

    A heurística procura vértices salientes acima do centro da palma e junta
    candidatos próximos. Depende de uma silhueta limpa e não reconhece gestos
    arbitrários, mãos rodadas ou dedos sobrepostos.
    """
    if contour is None or len(contour) < 3 or cv.contourArea(contour) <= 0:
        return 0, []

    x, y, width, height = cv.boundingRect(contour)
    local = contour.astype(np.int32, copy=True)
    local[:, 0, 0] -= x - 1
    local[:, 0, 1] -= y - 1
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv.drawContours(mask, [local], -1, 255, cv.FILLED)
    distances = cv.distanceTransform(mask, cv.DIST_L2, 5)
    _, radius, _, center = cv.minMaxLoc(distances)
    if radius < 2:
        return 0, []
    palm = np.array((center[0] + x - 1, center[1] + y - 1), dtype=float)

    polygon = cv.approxPolyDP(contour, 0.012 * cv.arcLength(contour, True), True)
    vertices = polygon.reshape(-1, 2).astype(float)
    orientation = cv.contourArea(polygon, oriented=True)
    candidates = []
    for index, point in enumerate(vertices):
        if point[1] >= palm[1] - 0.4 * radius:
            continue
        distance = float(np.linalg.norm(point - palm))
        if distance < 1.65 * radius:
            continue
        before = vertices[index - 1] - point
        after = vertices[(index + 1) % len(vertices)] - point
        denominator = np.linalg.norm(before) * np.linalg.norm(after)
        if denominator == 0:
            continue
        angle = math.degrees(math.acos(float(np.clip(np.dot(before, after) / denominator, -1, 1))))
        # Apenas vértices convexos; os vales entre dedos são côncavos.
        cross = before[0] * after[1] - before[1] * after[0]
        if angle <= 95 and cross * orientation < 0:
            candidates.append((distance, point))

    tips = []
    for _, point in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(np.linalg.norm(point - np.array(tip)) >= 0.55 * radius for tip in tips):
            tips.append((int(point[0]), int(point[1])))
        if len(tips) == 5:
            break
    tips.sort()
    return len(tips), tips
