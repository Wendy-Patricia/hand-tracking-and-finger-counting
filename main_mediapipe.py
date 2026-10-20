
"""
Application OpenCV + MediaPipe.

Partie A : segmentation HSV et detection du contour.
Partie B : squelette MediaPipe et comptage des doigts.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2 as cv

from controls import (
    WINDOW,
    create_trackbars,
    read_parameters,
    reset_trackbars,
)
from pipeline import process_frame
from mediapipe_geometry import (
    MediaPipeHandAnalyzer,
    draw_skeleton,
)


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "hand_landmarker.task"
)

VIDEO_WINDOW = "Detection et squelette MediaPipe"
MASK_WINDOW = "Masque binaire"


def main():
    parser = argparse.ArgumentParser(
        description="Prototype OpenCV et MediaPipe."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        print(
            f"Erreur : modele introuvable : {MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    capture = cv.VideoCapture(args.camera)

    if not capture.isOpened():
        print(
            "Erreur : impossible d'ouvrir la webcam.",
            file=sys.stderr,
        )
        return 1

    analyzer = None

    try:
        analyzer = MediaPipeHandAnalyzer(MODEL_PATH)

        cv.namedWindow(VIDEO_WINDOW, cv.WINDOW_NORMAL)
        cv.namedWindow(MASK_WINDOW, cv.WINDOW_NORMAL)
        create_trackbars()

        print("Prototype OpenCV + MediaPipe")
        print("Q/Echap : quitter | R : reinitialiser")

        start_time = time.monotonic()

        while True:
            ok, frame = capture.read()

            if not ok or frame is None:
                raise RuntimeError(
                    "Impossible de lire l'image de la webcam."
                )

            frame = cv.flip(frame, 1)

            # PARTIE A : segmentation et contour.
            parameters = read_parameters()
            result_a = process_frame(frame, parameters)

            display = frame.copy()

            if result_a.detection is not None:
                cv.drawContours(
                    display,
                    [result_a.detection.contour],
                    -1,
                    (0, 200, 0),
                    2,
                )

            # PARTIE B : squelette et comptage.
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result_b = analyzer.analyze(
                frame,
                timestamp_ms,
            )

            if result_b is not None:
                draw_skeleton(
                    display,
                    result_b["landmarks"],
                )

                cv.putText(
                    display,
                    f"Doigts estimes : {result_b['count']}",
                    (15, 35),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv.putText(
                    display,
                    "Squelette MediaPipe : 21 points",
                    (15, 67),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            else:
                cv.putText(
                    display,
                    "Aucune main detectee par MediaPipe",
                    (15, 35),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            cv.putText(
                display,
                "Q/Echap : quitter | R : reinitialiser",
                (15, display.shape[0] - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            cv.imshow(VIDEO_WINDOW, display)
            cv.imshow(MASK_WINDOW, result_a.mask)

            key = cv.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")):
                reset_trackbars()

            if any(
                cv.getWindowProperty(
                    name, cv.WND_PROP_VISIBLE
                ) < 1
                for name in (
                    VIDEO_WINDOW,
                    MASK_WINDOW,
                    WINDOW,
                )
            ):
                break

        return 0

    except (RuntimeError, cv.error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    finally:
        if analyzer is not None:
            analyzer.close()

        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
