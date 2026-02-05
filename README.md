# TP 5.1 — Réseaux de neurones récurrents (RNN) + GloVe

Ce dépôt contient les scripts correspondant au cahier Jupyter du **TP 5.1** :
- **Exercice 1** : génération de texte (poésie de Baudelaire) avec un RNN caractère-par-caractère  
- **Exercice 2** : embeddings **GloVe** sur les légendes **Flickr8k** (extraction + clustering + t-SNE)

---

## Structure du projet

```text
TP5_1_RNN_fleurs_du_mal/
├─ data/
│  ├─ fleurs_mal.txt
│  ├─ glove.6B.100d.txt
│  ├─ flickr_8k_train_dataset.txt
├─ exo0.py
├─ exo1.py
├─ exo2.py
├─ exo3.py
├─ exo4.py
├─ make_all.sh
└─ README.md


## Prérequis

Python 3.8+ recommandé.

Packages nécessaires :
- numpy
- pandas
- scikit-learn
- matplotlib
- keras / tensorflow

Installation (exemple) :
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow keras
