# Mini-projet CNN - Integration de TP1 a TP5

## 1. Vue d'ensemble

Ce depot contient un projet de reseaux de neurones convolutionnels (CNN)
concu pour integrer, dans un seul cadre experimental, les notions vues dans le cours de Online-Learning.  

Le projet courvre deux taches:

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


## 2. Structure actuelle du depot

```bash
.
├── README.md
├── mini_projet.pdf
├── pyproject.toml
├── uv.lock
├── .gitignore
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

### Role de fichiers

- `experiment_spec.py` : cadre experimental fige du projet.
- `data_loader.py` : dataset CelebA loader, transforms, et splits.
- `train_common.py` : briques communes d'entrainement, evaluation et sauvegarde.
- `train_classification.py` : entrainement de la classification binaire.
- `train_regression.py` : entrainement de la regression.
- `utils.py` : fonctions utilitaires reliees aux TPs (`covering number`, `line search`, `regularisation`, `regret`, `normes`, etc.).
- `tp1_related.ipynb` a `tp5_related.ipynb` : un notebook par TP.
- `models/cnn_simple.py` : implementation de `CNN1`.
- `models/cnn_improved.py` : implementation de `CNN2`.
- `outputs/` : checkpoints, metriques, configurations et courbes sauvegardees.

## 3. Donnees attendues

Le code suppose que le dossier suivant
existe localement:

```text
data/
├── img_align_celeba_reduced/
├── list_attr_celeba.csv
├── identity_celeba_reduced.csv
└── list_eval_partition_reduced.csv
```
Ordre:  
1. remettre les donnees dans `data/`
2. executer completement `tp1_related.ipynb`
3. verifier que `train_classification.py` et `train_regression.py` tournent
4. lancer les notebooks   


## 4. Commandes utiles

### Installation de l'environnement

Avec `uv`:

```bash
uv sync
```


### Lancer la classification

```bash
.venv/bin/python train_classification.py
```

### Lancer la regression

```bash
.venv/bin/python train_regression.py
```


