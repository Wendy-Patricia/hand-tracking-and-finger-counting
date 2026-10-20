
"""
Application de test du comptage des doigts.

Utilisation :
    python finger_counter.py tests/hand_5.png

L'image doit contenir une silhouette blanche
de la main sur un fond noir.
"""

import argparse
import cv2 as cv

from hand_geometry import analyze_fingers
from visualization import draw_results


def load_contour(mask):
    """Extrait le contour principal d'un masque binaire."""

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    candidate = max(
        contours, key=cv.contourArea
    )

    if cv.contourArea(candidate) < 1000:
        return None

    return candidate


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyse geometrique et comptage "
            "des doigts d'une main."
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Chemin vers une silhouette binaire "
            "de main (blanc sur noir)."
        )
    )

    args = parser.parse_args()

    # 1. Charger le masque
    mask = cv.imread(
        args.image,
        cv.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise SystemExit(
            "Erreur : image introuvable : " + args.image
        )

    # 2. Garantir un masque binaire
    _, mask = cv.threshold(
        mask,
        127,
        255,
        cv.THRESH_BINARY
    )

    # 3. Extraire le contour
    contour = load_contour(mask)

    # 4. Analyser la geometrie
    analysis = analyze_fingers(contour)

    # 5. Afficher les informations
    print("\n===== RESULTATS DE L'ANALYSE =====")

    print(
        "Main detectee :",
        "Oui" if analysis["detected"] else "Non"
    )

    print(
        "Doigts estimes :",
        analysis["count"]
    )

    print(
        "Creux detectes :",
        len(analysis["valleys"])
    )

    print(
        "Extremites detectees :",
        len(analysis["fingertips"])
    )

    print(
        "Pouce detecte :",
        "Oui" if analysis["thumb"] else "Non"
    )

    print(
        "Fiabilite :",
        "Indication favorable"
        if analysis["reliable"]
        else "Comptage incertain"
    )

    # 6. Preparer l'image couleur
    frame = cv.cvtColor(
        mask,
        cv.COLOR_GRAY2BGR
    )

    # 7. Dessiner les resultats
    output = draw_results(
        frame, contour, analysis
    )

    # 8. Afficher la fenetre
    cv.imshow(
        "Analyse geometrique de la main",
        output
    )

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
