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

Pour sélectionner une autre webcam :

```powershell
.\.venv\Scripts\python.exe camera.py --camera 1
```

## Utilisation et réglages

1. Placez une seule main devant un fond uni, de couleur différente de votre peau.
2. Présentez la paume face à la caméra, avec le poignet vers le bas.
3. Dans la fenêtre « Reglages HSV », ajustez les curseurs pour obtenir une main blanche et un fond noir dans le masque binaire.
4. Écartez les doigts levés pour que leurs silhouettes restent distinctes.
5. Attendez quelques images pour que le comptage stabilisé apparaisse.

| Réglage | Fonction |
| --- | --- |
| H min / H max | Intervalle de teinte, entre 0 et 179. Si le minimum dépasse le maximum, l'intervalle traverse la limite des teintes rouges. |
| S min / S max | Intervalle de saturation, entre 0 et 255. Les bornes inversées sont remises dans l'ordre. |
| V min / V max | Intervalle de luminosité, entre 0 et 255. Les bornes inversées sont remises dans l'ordre. |
| Flou | Taille du noyau gaussien : `2 × valeur + 1`. |
| Erosions | Réduit les zones blanches et les petits éléments parasites. |
| Dilatations | Agrandit les zones blanches. Une valeur trop élevée peut fusionner les doigts. |
| Aire min | Aire minimale du contour retenu, en pixels carrés. |

Les valeurs initiales doivent être adaptées à l'éclairage et au fond.

- **R** : réinitialiser les réglages et l'historique du comptage.
- **Q** ou **Échap** : quitter.
- Fermer une fenêtre termine également l'application.

L'affichage présente la mesure instantanée et le nombre stabilisé. La mention
« en attente » apparaît avant l'obtention d'un résultat stabilisé. La mention
« Comptage incertain » signale une mesure non validée par les critères géométriques.
Les extrémités sont rouges, les creux orange et le point de référence de la paume
magenta. Les articulations du squelette sont estimées par interpolation.

## Méthode de comptage

1. Convertir l'image de BGR vers HSV et construire le masque binaire.
2. Filtrer le masque et sélectionner son plus grand contour externe admissible.
3. Estimer la paume par transformée de distance, en limitant l'influence de l'avant-bras.
4. Rééchantillonner le contour à intervalles réguliers et le lisser.
5. Rechercher les pics de distance à la paume, puis vérifier leur proéminence et
   leur courbure locale pour retenir les extrémités des doigts.
6. Fusionner les candidats trop proches et rechercher les creux par défauts de convexité.
7. Stabiliser le nombre par vote sur les **7 dernières images**, avec **4 votes**
   nécessaires pour valider un nouveau nombre.

Les seuils de longueur sont proportionnels au rayon de la paume. Le pouce peut
être latéral et une silhouette compacte sans extrémité retenue peut être comptée
comme un poing fermé, soit zéro doigt.

Les images incertaines font vieillir l'historique : un ancien nombre ne reste pas
indéfiniment affiché. L'absence de main réinitialise immédiatement la stabilisation.

## Organisation du projet

| Fichier | Rôle |
| --- | --- |
| `camera.py` | Application webcam : acquisition, comptage stabilisé et affichage. |
| `config.py` | Paramètres typés du traitement dans `ProcessingParameters`. |
| `controls.py` | Création, lecture et réinitialisation des curseurs. |
| `segmentation.py` | Segmentation HSV, flou, seuillage et morphologie. |
| `hand_detection.py` | Sélection du contour, enveloppe convexe et description de forme. |
| `hand_geometry.py` | Analyse géométrique de la paume, des extrémités et des creux. |
| `temporal_filter.py` | Stabilisation du comptage avec `CountStabilizer`. |
| `pipeline.py` | Traitement d'une image sans interface graphique. |
| `visualization.py` | Dessin de la silhouette, des doigts, du comptage et du squelette. |
| `skeleton_geometry.py` | Estimation des points et connexions du squelette. |
| `finger_counter.py` | Analyse interactive d'une silhouette depuis un fichier. |
| `tests/` | Tests automatisés sans webcam physique. |

## Traiter une image dans du code

```python
import cv2 as cv

from config import ProcessingParameters
from pipeline import process_frame

image = cv.imread("main.jpg")
if image is None:
    raise ValueError("Image introuvable")

resultat = process_frame(image, ProcessingParameters(min_area=2000))
analyse = resultat.analysis

if analyse["detected"]:
    print("Nombre de doigts :", analyse["count"])
    print("Mesure géométriquement plausible :", analyse["reliable"])
    print("Extrémités :", analyse["fingertips"])
else:
    print("Aucune silhouette exploitable")
```

`FrameResult` contient le masque binaire (`mask`), la détection (`detection`, ou
`None`) et l'analyse des doigts (`analysis`). Les coordonnées sont exprimées dans
le repère de l'image source. `process_frame` ne modifie pas cette image et ne
conserve aucun historique. La stabilisation est appliquée dans l'application
webcam à l'aide de `CountStabilizer`. Les fonctions de visualisation dessinent
sur une copie et préservent également l'image source.

Pour analyser une silhouette blanche sur fond noir depuis un fichier :

```powershell
.\.venv\Scripts\python.exe finger_counter.py chemin\vers\silhouette.png
```

## Tests automatisés

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Les **16 tests automatisés** vérifient notamment :

- Les intervalles HSV, la sélection des contours et le filtrage par aire.
- La conservation de l'image source et le fonctionnement sans interface graphique.
- Les 32 combinaisons de doigts sur des silhouettes synthétiques et leur miroir.
- Le comptage de 0 à 5 doigts à trois échelles et avec des rotations de −20°, 0° et +20°.
- Le pouce seul, le poing fermé et l'influence d'un avant-bras long.
- Les contours simplifiés, translatés ou dont le point de départ change.
- La stabilisation, le passage à zéro et l'expiration des résultats anciens.
- L'application webcam avec des images simulées et la libération des ressources.

Ces tests ne mesurent pas la précision sur des images de mains réelles.
Les tests génèrent leurs propres silhouettes et ne nécessitent aucun fichier
d'image externe.

## Vérification avec une webcam

Après avoir réglé le masque, effectuez les essais suivants :

| Essai | Points à vérifier |
| --- | --- |
| Montrer successivement 0 à 5 doigts | Cohérence des extrémités et du nombre stabilisé, y compris pour le poing. |
| Lever uniquement le pouce, puis changer de main | Détection du doigt latéral des deux côtés. |
| Incliner légèrement la main | Maintien du comptage tant que le poignet reste orienté vers le bas. |
| Rapprocher puis éloigner la main | Stabilité du comptage ; adapter l'aire minimale si nécessaire. |
| Changer de fond ou d'éclairage | Main blanche et fond noir dans le masque ; réajuster HSV si nécessaire. |
| Retirer la main du champ | Disparition de la détection et réinitialisation du comptage. |

## Limites

La sélection repose sur la couleur et la géométrie, sans reconnaissance sémantique
de la main. Un visage, un bras ou un objet de couleur similaire peut être retenu.
Une seule silhouette est analysée : les mains multiples ne sont pas distinguées.

La paume doit faire face à la caméra et le poignet doit pointer approximativement
vers le bas. Une main tournée de 90° ou dirigée vers le bas n'est pas prise en charge.
Des doigts collés, masqués ou repliés derrière la paume ne peuvent pas être
comptés de manière fiable à partir de la silhouette seule.

Le champ `reliable` indique une plausibilité géométrique, pas une probabilité de
réussite. Les points du squelette sont interpolés et ne constituent pas une mesure
réelle des articulations. La qualité du résultat dépend du masque, du fond,
de l'éclairage et de la visibilité des doigts.
