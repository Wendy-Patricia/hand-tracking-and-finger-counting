# Détection de la main par segmentation HSV

Application Python autonome utilisant OpenCV et la webcam. Aucun modèle ni service
externe n'est nécessaire. Les images sont traitées localement, sans enregistrement.

## Installation et lancement

Avec Python 3.10 ou supérieur :

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python camera.py
```

Pour sélectionner une autre webcam : ajoutez `--camera 1`.
Une installation OpenCV avec interface graphique est nécessaire ;
`opencv-python-headless` ne permet pas d'afficher les fenêtres et les curseurs.

## Organisation

- `camera.py` : point d'entrée, capture webcam, affichage miroir, boucle et fermeture
  des ressources.
- `config.py` : paramètres typés et valeurs par défaut (`ProcessingParameters`),
  indépendants des libellés de l'interface.
- `controls.py` : création, lecture et réinitialisation des curseurs OpenCV ;
  conversion des valeurs de l'interface vers `ProcessingParameters`.
- `segmentation.py` : conversion BGR → HSV, masque par intervalles, flou gaussien,
  seuillage binaire, érosion et dilatation.
- `hand_detection.py` : recherche des contours avec `cv.findContours`, sélection
  du plus grand contour, enveloppe convexe et rectangle délimitant ; résultat
  structuré dans `HandDetection`.
- `pipeline.py` : enchaînement segmentation → détection pour une image BGR ;
  renvoie un `FrameResult` (masque et détection) sans ouvrir de fenêtre.
- `rendering.py` : dessin des contours, de l'enveloppe, du rectangle et des textes.

Pour traiter une image depuis un fichier, une vidéo ou une autre interface :

```python
import cv2 as cv
from config import ProcessingParameters
from pipeline import process_frame

frame = cv.imread("main.jpg")
if frame is None:
    raise ValueError("Image introuvable")
result = process_frame(frame, ProcessingParameters(min_area=2000))
# result.mask : masque binaire ; result.detection : HandDetection ou None.
```

Le traitement ne modifie pas l'image source. `draw_detection` dans `rendering.py`
annote l'image reçue : lui passer `frame.copy()` pour conserver l'original.
Les fonctions de curseurs sont désormais dans `controls.py`, et `draw_detection`
dans `rendering.py`. `segment_hand` reçoit un `ProcessingParameters` à la place
du dictionnaire de libellés de curseurs.

## Tests automatisés

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Les tests utilisent des images synthétiques pour vérifier les intervalles HSV,
la sélection des contours, le filtrage par aire, le traitement sans interface
et la conservation de l'image source. La fermeture des ressources est vérifiée
avec une webcam simulée ; les essais physiques restent décrits ci-dessous.

## Utilisation

Placez une main devant un fond uni, de couleur différente de votre peau.
Dans « Reglages HSV », ajustez H (teinte, 0–179), S (saturation, 0–255) et
V (luminosité, 0–255) pour que la main apparaisse en blanc dans le masque
et le fond en noir. Les valeurs initiales sont un point de départ à adapter.
Si H min dépasse H max, l'intervalle traverse la limite des teintes rouges.
Les bornes S et V sont automatiquement remises dans l'ordre.

Le curseur « Flou » définit un noyau de taille `2 × valeur + 1`.
« Erosions » supprime les petits éléments et « Dilatations » agrandit les zones
blanches ; des valeurs excessives peuvent fusionner les doigts ou les objets.
« Aire min » filtre les petits contours (en pixels carrés).
Appuyez sur **R** pour réinitialiser les réglages, **Q** ou **Échap** pour quitter.
Fermer une fenêtre termine également l'application.

Le contour est vert, l'enveloppe convexe bleue et le rectangle jaune.
La forme est décrite de façon indicative : compacte, allongée ou découpée.
La solidité est le rapport entre l'aire du contour et celle de son enveloppe
convexe. Une faible solidité peut correspondre à des doigts écartés.
Cette heuristique ne compte pas les doigts et ne reconnaît pas des gestes.

## Protocole de test avec la webcam

Effectuez chaque essai après avoir réglé le masque, puis notez la stabilité
du contour et les éventuelles zones du fond détectées à tort.

| Essai | Variations | Points à vérifier |
| --- | --- | --- |
| Positions | Paume ouverte, poing fermé, doigts écartés, rotation | Rectangle et enveloppe suivent la silhouette ; description indicative cohérente |
| Distance | Main proche puis éloignée | Adapter l'aire minimale si la main devient trop petite |
| Fonds | Fond uni clair, uni sombre, puis chargé | Repérer les objets de couleur similaire sélectionnés |
| Éclairage | Lumière naturelle, artificielle, faible, latérale | Réajuster S et V ; observer les ombres et les reflets |
| Absence de main | Retirer la main du champ | Aucun contour candidat sur un fond correctement exclu |

La méthode choisit le plus grand objet correspondant aux couleurs : un visage,
un bras ou un objet similaire peut donc être sélectionné. Elle ne distingue pas
plusieurs mains. Les essais physiques de position, de fond et de lumière doivent
être réalisés sur votre webcam ; leur résultat dépend de votre environnement.
