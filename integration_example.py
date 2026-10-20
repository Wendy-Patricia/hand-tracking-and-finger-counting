
"""
Interface d'integration du module B.

Le module A fournit :
    frame : image couleur actuelle
    contour : contour de la main ou None

Le module B effectue :
    analyse geometrique
    comptage des doigts
    stabilisation temporelle
    visualisation
"""

import cv2 as cv

from hand_geometry import analyze_fingers
from visualization import draw_results
from temporal_filter import CountStabilizer


class HandAnalysisProcessor:
    """Traitement reutilisable pour une sequence video."""

    def __init__(self):
        self.stabilizer = CountStabilizer(
            window=7,
            min_votes=4
        )

    def process_frame(self, frame, contour):
        """
        Analyse un frame et retourne
        l'image annotee et les resultats.
        """

        analysis = analyze_fingers(contour)

        stable_count = self.stabilizer.update(
            analysis
        )

        output = draw_results(
            frame,
            contour,
            analysis
        )

        height, width = output.shape[:2]

        # Bandeau inferieur pour le comptage stabilise
        cv.rectangle(
            output,
            (0, height - 45),
            (width, height),
            (25, 25, 25),
            -1
        )

        if stable_count is None:
            label = "Comptage stabilise : indisponible"
            color = (0, 165, 255)
        else:
            label = f"Comptage stabilise : {stable_count}"
            color = (0, 255, 0)

        cv.putText(
            output,
            label,
            (15, height - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        return output, analysis, stable_count

    def reset(self):
        """Reinitialise la stabilisation."""
        self.stabilizer.reset()
