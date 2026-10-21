
"""Stabilisation du comptage par vote sur une fenêtre d'images."""

from collections import Counter, deque


class CountStabilizer:
    """Conserve le dernier nombre confirmé et laisse expirer les mesures anciennes."""

    def __init__(self, window=7, min_votes=4):
        if window < 1:
            raise ValueError("La taille de la fenêtre doit être positive.")

        if not 1 <= min_votes <= window:
            raise ValueError("Le nombre de votes est invalide.")

        self.history = deque(maxlen=window)
        self.min_votes = min_votes
        self.stable_count = None

    def update(self, analysis):
        """Renvoie le nombre stabilisé, ou None si aucun résultat n'est disponible."""

        if not analysis.get("detected", False):
            self.reset()
            return None

        count = analysis.get("count")
        valid = (
            analysis.get("reliable", False)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= 5
        )
        # Les images incertaines font aussi vieillir les mesures précédentes.
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
        """Réinitialise l'historique et le comptage."""
        self.history.clear()
        self.stable_count = None
