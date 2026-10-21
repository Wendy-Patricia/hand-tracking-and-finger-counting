
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

        count = analysis.get("count")
        valid = (analysis.get("reliable", False)
                 and isinstance(count, int) and not isinstance(count, bool)
                 and 0 <= count <= 5)
        # Uncertain frames also age the window, so a stale count cannot
        # remain visible indefinitely after the silhouette deteriorates.
        self.history.append(count if valid else None)
        votes = Counter(value for value in self.history if value is not None)
        if not votes:
            self.stable_count = None
            return None

        candidate, frequency = votes.most_common(1)[0]

        if frequency >= self.min_votes:
            self.stable_count = candidate

        return self.stable_count

    def reset(self):
        """Reinitialise l'historique et le comptage."""
        self.history.clear()
        self.stable_count = None
