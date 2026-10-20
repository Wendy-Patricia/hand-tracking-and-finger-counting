
"""
Stabilisation temporelle du comptage des doigts.

Module B - Traitement des resultats image par image.
"""

from collections import Counter, deque


class CountStabilizer:
    """Stabilise le nombre de doigts sur plusieurs images."""

    def __init__(self, window=7, min_votes=4):
        if window < 1:
            raise ValueError(
                "La taille de la fenetre doit etre positive."
            )

        if not 1 <= min_votes <= window:
            raise ValueError(
                "Le nombre de votes est invalide."
            )

        self.history = deque(maxlen=window)
        self.min_votes = min_votes
        self.stable_count = None

    def update(self, analysis):
        """
        Met a jour la stabilisation.

        Retourne :
            int : nombre stabilise
            None : resultat indisponible
        """

        if not analysis.get("detected", False):
            self.reset()
            return None

        if not analysis.get("reliable", False):
            # Ne pas ajouter de mesure incertaine
            return self.stable_count

        count = analysis.get("count")

        if not isinstance(count, int) or not 0 <= count <= 5:
            return self.stable_count

        self.history.append(count)

        votes = Counter(self.history)

        candidate, frequency = votes.most_common(1)[0]

        if frequency >= self.min_votes:
            self.stable_count = candidate

        return self.stable_count

    def reset(self):
        """Reinitialise l'historique et le comptage."""
        self.history.clear()
        self.stable_count = None
