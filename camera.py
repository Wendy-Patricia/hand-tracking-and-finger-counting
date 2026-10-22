"""Détection et comptage des doigts avec une webcam : python camera.py."""

import argparse
import sys

import cv2 as cv

from controls import WINDOW, create_trackbars, read_parameters, reset_trackbars
from pipeline import process_frame
from temporal_filter import CountStabilizer
from visualization import draw_detection, draw_results


VIDEO_WINDOW = "Detection et comptage des doigts"
MASK_WINDOW = "Masque binaire"
WINDOWS = (VIDEO_WINDOW, MASK_WINDOW, WINDOW)
HELP_TEXT = "Q/Echap : quitter | R : reinitialiser"


def main(argv=None):
    """Lance la webcam et libère les ressources, même en cas d'erreur."""
    parser = argparse.ArgumentParser(description="Détection et comptage des doigts.")
    parser.add_argument(
        "--camera", type=int, default=0, help="Indice de la webcam (défaut : 0)"
    )
    args = parser.parse_args(argv)
    capture = cv.VideoCapture(args.camera)
    stabilizer = CountStabilizer()

    try:
        if not capture.isOpened():
            raise RuntimeError(
                "Impossible d'ouvrir la webcam. Vérifiez sa connexion, "
                "les autorisations ou essayez --camera 1."
            )

        cv.namedWindow(VIDEO_WINDOW, cv.WINDOW_NORMAL)
        cv.namedWindow(MASK_WINDOW, cv.WINDOW_NORMAL)
        create_trackbars()
        print("Réglez HSV pour isoler la main dans le masque blanc.")
        print(HELP_TEXT)

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("La webcam ne fournit plus d'image.")

            frame = cv.flip(frame, 1)
            
            parameters = read_parameters()
            result = process_frame(frame, parameters)
            
            display = draw_detection(frame.copy(), result.detection)
            for tip in result.fingertips:
                cv.circle(display, tip, 6, (0, 0, 255), -1)
            cv.putText(display, f"Doigts : {result.count}", (15, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv.putText(display, "Q/Echap : quitter | R : reinitialiser", (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow(VIDEO_WINDOW, display)
            cv.imshow(MASK_WINDOW, result.mask)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                reset_trackbars()
                stabilizer.reset()
            if any(cv.getWindowProperty(name, cv.WND_PROP_VISIBLE) < 1 for name in WINDOWS):
                break

        return 0
    except (RuntimeError, cv.error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
