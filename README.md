# Mini-projet CNN - Integration de TP1 a TP5

## 1. Vue d'ensemble

Ce depot contient un mini-projet de reseaux de neurones convolutionnels (CNN)
concu pour integrer, dans un seul cadre experimental, les notions vues dans
TP1 a TP5.

Le projet s'appuie sur un jeu de donnees d'images de type CelebA et doit
couvrir deux taches:

- une tache de regression
- une tache de classification binaire

Le cadre experimental retenu actuellement est le suivant:

- **Dataset vise**: CelebA reduced
- **CNN1**: architecture simple
- **CNN2**: architecture amelioree
- **Regression**: prediction de la somme de tous les attributs disponibles
- **Classification binaire**: prediction de l'attribut `Smiling`
- **Formalisme classification aligne avec le PDF**:
  - labels en `{-1, +1}`
  - score brut du reseau
  - decision par `sign(score)`
  - perte de type hinge

Le document de reference du mini-projet est:

- `mini_projet.pdf`

Guide complementaire utile:

- [training_reuse_guide.md](./training_reuse_guide.md) pour savoir quand reutiliser un entrainement et quand re-entrainer selon TP1 a TP5

## 2. Objectif pedagogique

Le but n'est pas seulement d'entrainer un modele, mais de relier les concepts
theoriques des TPs a des experiences concretes sur deux architectures CNN
differentes.

Le projet final doit permettre de:

- comparer deux architectures CNN
- traiter une regression et une classification binaire
- analyser la complexite des donnees
- etudier gradient, sous-gradient, line search, regularisation
- comparer plusieurs optimiseurs
- introduire des variantes online et stochastiques
- produire des courbes, tableaux et conclusions argumentees

## 3. Structure actuelle du depot

Le projet conserve volontairement une structure simple, avec les fichiers
principaux a la racine.

```text
.
├── README.md
├── mini_projet.pdf
├── pyproject.toml
├── uv.lock
├── .gitignore
├── main.py
├── experiment_spec.py
├── data_loader.py
├── train_common.py
├── train_classification.py
├── train_regression.py
├── training_reuse_guide.md
├── utils.py
├── tp1_related.ipynb
├── tp2_related.ipynb
├── tp3_related.ipynb
├── tp4_related.ipynb
├── tp5_related.ipynb
├── outputs/
│   ├── classification/
│   └── regression/
└── models/
    ├── __init__.py
    ├── cnn_simple.py
    └── cnn_improved.py
```

### Role de chaque fichier

- `README.md` : vue d'ensemble, structure, commandes et avancement.
- `mini_projet.pdf` : sujet officiel de reference.
- `pyproject.toml` / `uv.lock` : dependances et verrouillage d'environnement.
- `.gitignore` : ignore `data/`, caches et fichiers temporaires.
- `main.py` : point d'entree principal (`status`, regression, classification).
- `experiment_spec.py` : cadre experimental fige du projet.
- `data_loader.py` : dataset CelebA, transforms, splits et `DataLoader`.
- `train_common.py` : briques communes d'entrainement, evaluation et sauvegarde.
- `train_classification.py` : entrainement de la classification binaire.
- `train_regression.py` : entrainement de la regression.
- `training_reuse_guide.md` : guide sur la reutilisation ou non des entrainements selon TP1 a TP5.
- `utils.py` : fonctions utilitaires reliees aux TPs (`covering number`, line search, regularisation, regret, normes, etc.).
- `tp1_related.ipynb` a `tp5_related.ipynb` : un notebook par TP, aligne sur le PDF.
- `models/cnn_simple.py` : implementation de `CNN1`.
- `models/cnn_improved.py` : implementation de `CNN2`.
- `outputs/` : checkpoints, metriques, configurations et courbes sauvegardees.

## 4. Donnees attendues

Le repo ne versionne pas les donnees. Le code suppose que le dossier suivant
existe localement:

```text
data/
├── img_align_celeba_reduced/
├── list_attr_celeba_reduced.csv
├── list_attr_celeba.csv
├── identity_celeba_reduced.csv
└── list_eval_partition_reduced.csv
```

Selon les scripts, le projet peut aussi retomber sur:

- `data/img_align_celeba`
- `data/list_attr_celeba.csv`
- `data/list_eval_partition.csv`

Sans le dossier `data/`, les scripts d'entrainement et le notebook ne peuvent
pas aller jusqu'au bout.

## 5. Commandes utiles

### Installation de l'environnement

Avec `uv`:

```bash
uv sync
```

Si besoin, une version CPU de PyTorch peut etre installee via:

```bash
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

### Voir l'etat du projet

```bash
python3 main.py status
```

ou avec l'environnement du projet:

```bash
.venv/bin/python main.py status
```

### Lancer la classification

```bash
.venv/bin/python train_classification.py
```

### Lancer la regression

```bash
.venv/bin/python train_regression.py
```

## 6. Avancement actuel

### Ce qui est deja fait

- Le sujet du mini-projet a ete lu et compare au code existant.
- Un cadre experimental explicite a ete fixe dans `experiment_spec.py`.
- Deux architectures CNN sont en place:
  - `CNN1` simple
  - `CNN2` improved
- Les scripts d'entrainement existent pour:
  - la classification
  - la regression
- La classification a ete recalee sur le formalisme du PDF:
  - score brut
  - labels signes `{-1, +1}`
  - perte hinge
- Le notebook `tp1_related.ipynb` a ete complete pour mieux couvrir TP1:
  - exploration
  - covering number
  - bruit / valeurs atypiques / densite
  - gradient differentiable
  - line search
  - biais-variance
  - validation et regularisation
- Les notebooks `tp2_related.ipynb`, `tp3_related.ipynb`, `tp4_related.ipynb`
  et `tp5_related.ipynb` ont ete ajoutes et structures selon le PDF.
- Les scripts d'entrainement sauvegardent maintenant les artefacts dans
  `outputs/`:
  - checkpoint
  - metriques
  - configuration
  - courbe de loss
- `torch` et les dependances principales ont ete installes dans la `.venv`.

### Ce qui bloque encore

- Le dossier `data/` est absent du repo local au moment de la redaction de ce
  README.
- A cause de cela, les cellules d'entrainement du notebook et les scripts ne
  peuvent pas etre executes jusqu'au bout.

### Ce qui reste a faire

- Ajouter les donnees dans `data/`
- Executer proprement les notebooks `tp1_related.ipynb` a `tp5_related.ipynb`
- Verifier et corriger les cellules si des erreurs runtime apparaissent
- Lancer les scripts d'entrainement complets sur les vraies donnees
- Produire les tableaux, figures et conclusions finales
- Rediger le rapport final

## 7. Proposition de progression

Ordre conseille pour la suite:

1. remettre les donnees dans `data/`
2. executer completement `tp1_related.ipynb`
3. verifier que `train_classification.py` et `train_regression.py` tournent
4. creer un notebook par TP en gardant la meme structure simple
5. centraliser les resultats dans `outputs/` si besoin
6. ecrire le rapport final a partir des notebooks

## 8. Idee directrice du projet

Le projet doit rester lisible et pedagogique.

La priorite n'est pas de construire une grosse architecture logicielle, mais de
garder un depot simple ou l'on comprend rapidement:

- quel fichier fait quoi
- quelle experience correspond a quel TP
- quelles hypotheses ont ete choisies
- ou trouver les resultats

Dans cette logique, garder une structure simple avec:

- un script de regression
- un script de classification
- un notebook par TP
- un fichier de specification experimentale

est une bonne approche pour finir proprement le mini-projet.
