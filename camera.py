"""Application autonome de détection de main : python camera.py."""

import argparse
import sys

import cv2 as cv

from controls import WINDOW, create_trackbars, read_parameters, reset_trackbars
from pipeline import process_frame
from rendering import draw_detection
from temporal_filter import CountStabilizer
from visualization import draw_results


VIDEO_WINDOW = "Detection de la main"
MASK_WINDOW = "Masque binaire"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Détection de main par segmentation HSV.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la webcam (défaut : 0)")
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
        
        print("Réglez HSV pour isoler la main dans le masque blanc. " "Q/Échap : quitter ; R : réinitialiser.")
        
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("La webcam ne fournit plus d'image.")
            
            frame = cv.flip(frame, 1)
            
            parameters = read_parameters()
            result = process_frame(frame, parameters)
            
            analysis = result.analysis
            analysis['stable_count'] = stabilizer.update(analysis)
            if analysis['detected']:
                display = draw_results(frame, result.detection.contour, analysis)
            else:
                display = draw_detection(frame.copy(), result.detection)
            cv.putText(display, "Q/Echap : quitter | R : reinitialiser", (15, display.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow(VIDEO_WINDOW, display)
            cv.imshow(MASK_WINDOW, result.mask)
            key = cv.waitKey(1) & 0xFF
            
            if key in (27, ord("q"), ord("Q")):
                break
            if any(cv.getWindowProperty(name, cv.WND_PROP_VISIBLE) < 1
                   for name in (VIDEO_WINDOW, MASK_WINDOW, WINDOW)):
                break
            if key in (ord("r"), ord("R")):
                reset_trackbars()
                stabilizer.reset()
        return 0
    except (RuntimeError, cv.error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
