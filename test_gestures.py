
"""
Evaluation du comptage des doigts de 0 a 5.

Module B - Tests independants.

Les images doivent etre des masques binaires
avec une main blanche sur un fond noir.
"""

from pathlib import Path
import cv2 as cv

from hand_geometry import analyze_fingers
from visualization import draw_results


TEST_DIRECTORY = Path("tests")
OUTPUT_DIRECTORY = Path("results")


def extract_contour(mask):
    """Extrait le contour principal du masque."""

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv.contourArea)

    if cv.contourArea(contour) < 1000:
        return None

    return contour


def test_image(expected_count):
    """Teste une image et enregistre le resultat."""

    filename = f"hand_{expected_count}.png"
    image_path = TEST_DIRECTORY / filename

    mask = cv.imread(
        str(image_path),
        cv.IMREAD_GRAYSCALE
    )

    if mask is None:
        print(f"Image introuvable : {image_path}")
        return None

    _, mask = cv.threshold(
        mask, 127, 255, cv.THRESH_BINARY
    )

    contour = extract_contour(mask)

    analysis = analyze_fingers(contour)

    estimated = analysis["count"]

    correct = (
        analysis["detected"]
        and analysis["reliable"]
        and estimated == expected_count
    )

    # Creer l'image de visualisation
    frame = cv.cvtColor(
        mask, cv.COLOR_GRAY2BGR
    )

    result = draw_results(
        frame, contour, analysis
    )

    # Enregistrer la visualisation
    OUTPUT_DIRECTORY.mkdir(
        parents=True, exist_ok=True
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"result_hand_{expected_count}.png"
    )

    cv.imwrite(str(output_path), result)

    # Afficher les informations
    status = (
        "REUSSI" if correct else "A VERIFIER"
    )

    print(f"\n--- Test : {filename} ---")
    print(f"Nombre attendu : {expected_count}")
    print(f"Nombre estime : {estimated}")

    print(
        "Creux detectes :",
        len(analysis["valleys"])
    )

    print(
        "Extremites detectees :",
        len(analysis["fingertips"])
    )

    print(
        "Comptage fiable :",
        "Oui" if analysis["reliable"] else "Non"
    )

    print(f"Resultat : {status}")

    return correct


def main():

    print("===== EVALUATION DES GESTES =====")

    results = []

    for expected in range(6):
        result = test_image(expected)

        if result is not None:
            results.append(result)

    print("\n===== BILAN DES TESTS =====")

    if not results:
        print("Aucun test execute.")
        return

    successful = sum(results)
    total = len(results)

    print(f"Tests reussis : {successful}/{total}")

    print(
        f"Taux de reussite : "
        f"{successful / total * 100:.1f} %"
    )

    print(
        "Les visualisations sont "
        "enregistrees dans results/."
    )


if __name__ == "__main__":
    main()
