"""Analyse d'une silhouette blanche sur fond noir : python finger_counter.py image.png."""

import argparse

import cv2 as cv

from hand_detection import detect_hand
from hand_geometry import analyze_fingers
from visualization import draw_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comptage des doigts sur une silhouette.")
    parser.add_argument("image", help="Chemin du masque : main blanche sur fond noir.")
    args = parser.parse_args(argv)

    mask = cv.imread(args.image, cv.IMREAD_GRAYSCALE)
    if mask is None:
        parser.error(f"Image introuvable : {args.image}")

    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    detection = detect_hand(mask, min_area=1000)
    contour = detection.contour if detection is not None else None
    analysis = analyze_fingers(contour)

    print("Main détectée :", "oui" if analysis["detected"] else "non")
    print("Doigts estimés :", analysis["count"])
    print("Creux détectés :", len(analysis["valleys"]))
    print("Pouce détecté :", "oui" if analysis["thumb"] is not None else "non")
    print("Comptage :", "plausible" if analysis["reliable"] else "incertain")

    frame = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    try:
        cv.imshow("Analyse de la silhouette", draw_results(frame, contour, analysis))
        cv.waitKey(0)
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
