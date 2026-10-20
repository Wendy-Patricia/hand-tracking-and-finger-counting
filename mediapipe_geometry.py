
"""
Estimation du squelette et comptage des doigts avec MediaPipe.
"""

import math

import cv2 as cv
import mediapipe as mp
import numpy as np


# Connexions anatomiques des 21 points.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGERS = [
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
]


def distance(a, b):
    """Distance entre deux points 3D normalises."""
    return math.dist(
        (a.x, a.y, a.z),
        (b.x, b.y, b.z),
    )


def angle(a, b, c):
    """Calcule l'angle ABC en degres."""
    first = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    ])

    second = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z,
    ])

    norm = np.linalg.norm(first) * np.linalg.norm(second)

    if norm < 1e-8:
        return 0.0

    cosine = np.clip(
        np.dot(first, second) / norm,
        -1.0,
        1.0,
    )

    return math.degrees(math.acos(cosine))


def count_fingers(landmarks):
    """
    Compte approximativement les doigts etendus.

    Les seuils sont experimentaux.
    """

    count = 0

    # Index, majeur, annulaire et auriculaire.
    for mcp, pip, dip, tip in FINGERS:

        pip_angle = angle(
            landmarks[mcp],
            landmarks[pip],
            landmarks[dip],
        )

        dip_angle = angle(
            landmarks[pip],
            landmarks[dip],
            landmarks[tip],
        )

        extension = (
            pip_angle > 155
            and dip_angle > 150
            and distance(
                landmarks[tip],
                landmarks[mcp],
            ) > 1.5 * distance(
                landmarks[pip],
                landmarks[mcp],
            )
        )

        if extension:
            count += 1

    # Pouce.
    thumb_angle = angle(
        landmarks[2],
        landmarks[3],
        landmarks[4],
    )

    thumb_open = distance(
        landmarks[4],
        landmarks[5],
    ) > 1.15 * distance(
        landmarks[3],
        landmarks[5],
    )

    if thumb_angle > 150 and thumb_open:
        count += 1

    return count


def draw_skeleton(frame, landmarks):
    """Dessine les articulations et les connexions."""
    height, width = frame.shape[:2]

    points = [
        (
            int(point.x * width),
            int(point.y * height),
        )
        for point in landmarks
    ]

    # Connexions du squelette.
    for start_id, end_id in HAND_CONNECTIONS:

        cv.line(
            frame,
            points[start_id],
            points[end_id],
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )

    tips = {4, 8, 12, 16, 20}

    # Articulations.
    for index, point in enumerate(points):

        color = (
            (0, 0, 255)
            if index in tips
            else (0, 255, 255)
        )

        cv.circle(
            frame,
            point,
            5,
            color,
            -1,
            cv.LINE_AA,
        )

    return frame


class MediaPipeHandAnalyzer:
    """Analyse une seule main sur les images successives."""

    def __init__(self, model_path):
        vision = mp.tasks.vision

        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.last_timestamp = -1

    def analyze(self, frame, timestamp_ms):
        """Retourne les points et le nombre de doigts."""
        timestamp_ms = max(
            int(timestamp_ms),
            self.last_timestamp + 1,
        )

        self.last_timestamp = timestamp_ms

        rgb = cv.cvtColor(
            frame,
            cv.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        result = self.landmarker.detect_for_video(
            image,
            timestamp_ms,
        )

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]

        return {
            "landmarks": landmarks,
            "count": count_fingers(landmarks),
        }

    def close(self):
        """Libere les ressources MediaPipe."""
        self.landmarker.close()
