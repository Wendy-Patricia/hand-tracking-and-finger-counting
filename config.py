"""Paramètres du traitement, indépendants de l'interface graphique."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingParameters:
    h_min: int = 0
    h_max: int = 25
    s_min: int = 40
    s_max: int = 255
    v_min: int = 50
    v_max: int = 255
    blur: int = 3
    erosions: int = 1
    dilations: int = 2
    min_area: int = 3000
