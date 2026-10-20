
"""
Application integree de detection et d'analyse de la main.

Partie A : camera, segmentation HSV et detection.
Partie B : comptage des doigts et squelette geometrique.
"""

import argparse
import sys

import cv2 as cv

from controls import (
    WINDOW,
    create_trackbars,
    read_parameters,
    reset_trackbars,
)
from pipeline import process_frame
from rendering import draw_detection
from hand_geometry import analyze_fingers
from visualization import draw_results


VIDEO_WINDOW = "Detection et analyse de la main"
MASK_WINDOW = "Masque binaire"


def main():
    parser = argparse.ArgumentParser(
        description="Detection et comptage des doigts."
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    capture = cv.VideoCapture(args.camera)

    try:
        if not capture.isOpened():
            raise RuntimeError(
                "Impossible d'ouvrir la webcam."
            )

        cv.namedWindow(VIDEO_WINDOW, cv.WINDOW_NORMAL)
        cv.namedWindow(MASK_WINDOW, cv.WINDOW_NORMAL)

        create_trackbars()

        print("Detection et analyse de la main")
        print("Q/Echap : quitter | R : reinitialiser")

        while True:
            ok, frame = capture.read()

            if not ok or frame is None:
                raise RuntimeError(
                    "Impossible de lire l'image de la webcam."
                )

            frame = cv.flip(frame, 1)

            # Partie A : segmentation et detection
            parameters = read_parameters()
            result = process_frame(frame, parameters)

            # Image couleur pour les annotations
            display = frame.copy()

            if result.detection is not None:

                contour = result.detection.contour

                # Partie B : analyse des doigts
                analysis = analyze_fingers(contour)

                if analysis["detected"]:
                    # Comptage et squelette
                    display = draw_results(
                        display,
                        contour,
                        analysis
                    )
                else:
                    display = draw_detection(
                        display,
                        result.detection
                    )

            else:
                cv.putText(
                    display,
                    "Aucune main detectee",
                    (15, 70),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv.putText(
                display,
                "Q/Echap : quitter | R : reinitialiser",
                (15, display.shape[0] - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv.imshow(VIDEO_WINDOW, display)
            cv.imshow(MASK_WINDOW, result.mask)

            key = cv.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")):
                reset_trackbars()

        return 0

    except (RuntimeError, cv.error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
