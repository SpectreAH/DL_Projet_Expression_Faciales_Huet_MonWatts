# Reconnaissance et analyse automatique d'expressions faciales

Projet du module **Fondamentaux du Deep Learning** — CY Tech, 2026-2027.

Système de vision par ordinateur qui reconnaît l'expression d'un visage parmi
**8 classes** (neutre, joie, surprise, tristesse, colère, dégoût, peur, mépris),
puis l'étend à la **détection de plusieurs visages avec YOLO**, sur image, vidéo
et webcam, via une **application Streamlit**.

**En bref**

- CNN construit avec Keras sur le dataset **FERPlus** (31 412 images 48×48).
- **73,3 %** d'accuracy et **0,57** de F1 macro sur le test, contre 50,6 % pour un réseau dense.
- Démarche expérimentale rigoureuse : une variable modifiée à la fois, choix du modèle sur la validation, test évalué une seule fois.
- Pipeline complet : détection des visages (YOLOv8-face) → CNN → expression + probabilité, à environ **47 images/s sur CPU**.

## Équipe

Projet réalisé en binôme.

| Membre | GitHub |
|--------|--------|
| Daniel Mon Watts | [@damonwatts11](https://github.com/damonwatts11) |
| Alexandre Huet   | [@SpectreAH](https://github.com/SpectreAH) |

**Formatrice :** Hanane Zerdoum

## Résultats

| Modèle | Accuracy test | F1 macro test |
|--------|:-------------:|:-------------:|
| Réseau dense (référence) | 50,6 % | — |
| CNN de référence | 72,6 % | 0,44 |
| **Modèle final** (CNN, learning rate 1e-4) | **73,3 %** | **0,57** |

Le **F1 macro** (moyenne par classe) est notre critère principal, car le dataset
est très déséquilibré : l'accuracy seule favorise les classes majoritaires.

Comparaison des expériences (sur la **validation**) :

| # | Modification | Val. accuracy | Val. F1 macro |
|:-:|--------------|:-------------:|:-------------:|
| 0 | CNN de référence | 0,747 | 0,463 |
| 1 | Data augmentation | 0,771 | 0,493 |
| 2 | Poids des classes (`balanced`) | 0,350 | 0,090 |
| 3 | **Learning rate 1e-4** (retenu) | 0,750 | **0,586** |

**À retenir**

- La convolution apporte **+22 points** d'accuracy par rapport au réseau dense.
- Le modèle final reconnaît enfin une partie des classes rares (recall `dégoût` 0 → 0,22, `mépris` 0 → 0,19).
- L'expérience « poids des classes » a **échoué** : des poids extrêmes (26× pour `dégoût`/`mépris`) déstabilisent l'entraînement. Ce résultat négatif est conservé et expliqué dans le notebook.
- Confusions les plus fréquentes : `tristesse → neutre` et `peur → surprise`.

## Dataset — FER+ (FERPlus)

[FERPlus](https://github.com/microsoft/FERPlus) reprend les images de **FER-2013**
et remplace les labels par une **re-annotation à 10 votes** (plus fiable). Deux fichiers,
alignés ligne à ligne :

| Fichier | Contenu | Source |
|---------|---------|--------|
| `fer2013.csv` | pixels 48×48 en niveaux de gris (`emotion, pixels, Usage`) | Kaggle (FER-2013) |
| `fer2013new.csv` | votes des 8 émotions + `unknown`, `NF` | repo microsoft/FERPlus (inclus dans `data/`) |

- **35 887 images** ; après **vote majoritaire** (votes isolés écartés, plus de 50 % des
  votes restants exigés, `unknown`/`NF` retirés), il reste **31 412 images**.
- Découpage **officiel** via la colonne `Usage` : 25 060 train / 3 199 validation (`PublicTest`) / 3 153 test (`PrivateTest`).
- **Fort déséquilibre** : `neutre` + `joie` ≈ 64 % des images ; `dégoût` et `mépris` ≈ 0,5 % chacune.

## Démarche (notebook `notebooks/expressions_faciales.ipynb`)

| Partie | Contenu | Points clés |
|:------:|---------|-------------|
| 1 | Données et préparation | Chargement et alignement des 2 fichiers, labels par vote majoritaire, split officiel, normalisation `/255`, one-hot |
| 2 | Modèle de référence (dense) | `Flatten → Dense(256) → Softmax` : point de comparaison sans structure spatiale (50,6 %) |
| 3 | CNN | 3 blocs `Conv → BatchNorm → MaxPool` (32/64/128) → `Dense(128) + Dropout` → Softmax ; ~685k paramètres |
| 4 | Entraînement | Adam, `categorical_crossentropy`, EarlyStopping + ReduceLROnPlateau ; surapprentissage visible après l'époque 10 |
| 5 | Évaluation et erreurs | Matrice de confusion, résultats par classe, exemples d'erreurs |
| 6 | Expérimentations | 3 expériences (une variable à la fois), modèle final choisi sur le **F1 macro** de validation |
| 7 | Enrichissement | Data augmentation et régularisation, intégrées à la partie 6 |
| 8 | Détection multi-visages | **YOLOv8-face** → découpe des visages → même prétraitement → CNN |
| 9 | Vidéo (bonus) | Analyse image par image (~47 images/s sur CPU) |

Le jeu de **test** ne sert qu'à l'évaluation finale : tous les choix se font sur la **validation**.

## Application Streamlit

Démo interactive du modèle final :

- **Image** : téléverser une photo ou en prendre une avec la webcam → expression prédite + probabilités.
- **Vidéo / Webcam** : détection de plusieurs visages avec YOLOv8-face et prédiction en direct.

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

L'application charge `models/emotion_cnn.keras`, créé en exécutant le notebook
(voir [`app/README.md`](app/README.md)).

## Présentation

Les slides de la soutenance (9 octobre 2026) sont dans [`presentation/`](presentation/).
Le fichier est en HTML : téléchargez-le puis ouvrez-le dans un navigateur.

## Structure du dépôt

```
.
├── notebooks/
│   └── expressions_faciales.ipynb   # notebook complet (parties 0 à 9)
├── app/
│   ├── app.py                       # application Streamlit
│   ├── requirements.txt
│   └── README.md
├── presentation/                    # slides de la soutenance
├── data/
│   ├── fer2013new.csv               # labels FER+ (inclus)
│   └── fer2013.csv                  # pixels (non inclus : ~290 Mo, via Kaggle)
├── models/
│   └── yolov8n-face.pt              # détecteur de visages
├── requirements.txt
└── README.md
```

## Exécution

Le notebook télécharge les données automatiquement : `fer2013.csv` via **kagglehub**
(`deadskull7/fer2013`) et `fer2013new.csv` depuis le repo Microsoft. Un `kaggle.json`
peut être nécessaire pour la partie Kaggle.

### Google Colab (recommandé)

*Exécution → Modifier le type d'exécution → GPU*, puis **Exécuter tout**.

### En local

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/expressions_faciales.ipynb
```

## Limites et pistes d'amélioration

- Les classes rares (`dégoût`, `mépris`, `peur`) restent mal reconnues, faute d'exemples.
- Images petites (48×48) et en niveaux de gris.
- Pistes : combiner data augmentation et learning rate 1e-4, des poids de classes plus doux, le transfer learning avec un réseau pré-entraîné, et un lissage des prédictions dans le temps pour la vidéo.
