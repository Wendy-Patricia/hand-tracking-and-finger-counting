# Suivi de la main et comptage des doigts

Application Python de détection de la silhouette d'une main et de comptage de
**0 à 5 doigts**, en temps réel avec une webcam. Le traitement repose uniquement
sur **OpenCV et NumPy**, sans MediaPipe, réseau neuronal ni modèle préentraîné.

Les images sont traitées localement, sans enregistrement ni service externe.

## Fonctionnalités

- Segmentation de la main par réglage des intervalles de couleur HSV.
- Extraction du plus grand contour correspondant aux couleurs sélectionnées.
- Estimation du centre et du rayon de la paume.
- Comptage géométrique des doigts, y compris un pouce latéral et le poing fermé.
- Stabilisation temporelle pour limiter les variations du nombre affiché.
- Affichage des extrémités, des creux et d'un squelette géométrique estimé.
- Analyse d'images sans webcam ni fenêtre graphique.

## Installation

Prérequis : Python 3.10 ou supérieur et une webcam pour l'utilisation en direct.
Depuis le dossier du projet, sous Windows :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Les dépendances sont `numpy` et `opencv-python`. La version
`opencv-python-headless` ne convient pas aux applications avec fenêtres et curseurs.

## Lancement

`camera.py` est le point d'entrée de l'application webcam : détection,
comptage stabilisé et squelette estimé.

```powershell
.\.venv\Scripts\python.exe camera.py
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
- `hand_detection.py` : `detect_hand(frame)` reçoit une image BGR et renvoie
  le plus grand contour candidat (`np.ndarray`) ou `None`. `describe_hand`
  calcule séparément les annotations `HandDetection`.
- `finger_counting.py` : `count_fingers(contour)` renvoie `(count, fingertips)`,
  avec une liste de coordonnées `(x, y)` dans l'image d'origine.
- `pipeline.py` : enchaînement segmentation → détection pour une image BGR ;
  renvoie un `FrameResult` (masque, détection, compteur et pointes) sans fenêtre.
  Réutilise le masque pour éviter une deuxième segmentation.
- `rendering.py` : dessin des contours, de l'enveloppe, du rectangle et des textes.

Pour traiter une image depuis un fichier, une vidéo ou une autre interface :

Contrat minimal pour l'intégration entre les deux personnes :

```python
from hand_detection import detect_hand
from finger_counting import count_fingers

contour = detect_hand(frame)  # Pessoa A : frame BGR -> contour ou None
count, fingertips = count_fingers(contour)  # Pessoa B : contour -> int, liste
# Sans main : contour vaut None ; le compteur renvoie (0, []).
```

`detect_hand(frame, parameters)` accepte aussi les réglages facultatifs
`ProcessingParameters` ; l'appel minimal utilise les valeurs par défaut.
Les deux fonctions conservent leurs entrées sans les modifier.

Pour conserver également le masque et les annotations :

```python
import cv2 as cv

from config import ProcessingParameters
from pipeline import process_frame

image = cv.imread("main.jpg")
if image is None:
    raise ValueError("Image introuvable")
result = process_frame(frame, ProcessingParameters(min_area=2000))
# result.mask : masque binaire ; result.detection : HandDetection ou None.
# result.contour : contour ou None ; result.count : nombre de doigts.
# result.fingertips : liste des coordonnées (x, y) des pointes.
```

Le traitement ne modifie pas l'image source. `draw_detection` dans `rendering.py`
annote l'image reçue : lui passer `frame.copy()` pour conserver l'original.
Les fonctions de curseurs sont désormais dans `controls.py`, et `draw_detection`
dans `rendering.py`. `segment_hand` reçoit un `ProcessingParameters` à la place
du dictionnaire de libellés de curseurs.

## Tests automatisés

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
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
Le nombre de doigts estimé et les pointes rouges proviennent d'une heuristique
géométrique distincte. Elle suppose une paume visible avec les doigts vers le
haut et une silhouette propre ; les rotations, doigts joints et occultations
peuvent produire une estimation incorrecte. Elle ne reconnaît pas des gestes.

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
