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

---

# Complément — Mini-projet R516 : thématique 2

## Détection des mains, estimation du squelette et comptage des doigts

Ce complément présente la **partie B**, développée en parallèle de la détection OpenCV décrite ci-dessus. **Les sections précédentes correspondent au README original de la partie A et sont conservées telles quelles.**

La version expérimentale combine les deux traitements sur le même flux vidéo :

- **Partie A (OpenCV)** : acquisition webcam, segmentation HSV, masque binaire et extraction du plus grand contour (`config.py`, `controls.py`, `segmentation.py`, `hand_detection.py`, `pipeline.py`). Le programme autonome `camera.py` et son affichage `rendering.py` sont conservés pour démontrer cette partie séparément.
- **Partie B (MediaPipe)** : estimation de 21 repères de la main et comptage *approximatif* des doigts tendus à partir d'angles et de distances (`mediapipe_geometry.py`).
- **Intégration** : `main_mediapipe.py` exécute le traitement HSV et MediaPipe en parallèle et affiche le contour OpenCV avec le squelette et le nombre de doigts estimé.

**MediaPipe n'est utilisé que dans la variante expérimentale, sous réserve de l'autorisation de l'enseignant.** Il effectue sa propre détection sur l'image couleur : l'analyse MediaPipe ne dépend pas du masque HSV, même si les deux résultats sont affichés ensemble.

## Installation de la variante intégrée (macOS)

Utiliser **Python 3.12** avec un environnement dédié, depuis la racine du dépôt :

```bash
python3.12 -m venv .venv-mediapipe
source .venv-mediapipe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mediapipe.txt
```

Télécharger le modèle officiel Hand Landmarker (non inclus dans l'archive) :

```bash
mkdir -p models
curl -L --fail -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

Démarrer la version intégrée :

```bash
python main_mediapipe.py
```

Pour utiliser une autre caméra : `python main_mediapipe.py --camera 1`.
Les fenêtres comprennent le flux annoté, le masque binaire et les curseurs HSV. **Q/Échap** quitte ; **R** réinitialise les réglages.

La version autonome de la partie A reste disponible :

```bash
python camera.py
```

## Fichiers propres à la partie B

| Fichier | Rôle |
| --- | --- |
| `mediapipe_geometry.py` | Création du détecteur, estimation des 21 repères, comptage heuristique, dessin du squelette |
| `main_mediapipe.py` | Exécution simultanée du pipeline HSV de la partie A et de l'analyse MediaPipe |
| `requirements-mediapipe.txt` | Dépendances Python pour la variante expérimentale |
| `models/README.md` | Rappel du modèle à télécharger |

Les anciens scripts de comptage et de squelette **exclusivement OpenCV** ont été retirés de cette branche pour éviter les duplications ; ils restent dans l'historique Git et les autres branches du projet. Aucune fonctionnalité MediaPipe ne doit être annoncée comme faisant partie des méthodes vues en cours.

## Vérification et limites

Tests du traitement de la partie A, sans webcam :

```bash
python -m unittest discover -s tests -v
```


