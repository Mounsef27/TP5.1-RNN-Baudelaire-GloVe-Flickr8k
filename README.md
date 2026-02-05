# TP 5.1 — Réseaux de neurones récurrents (RNN) + GloVe

- **Exercice 1** : génération de texte (Baudelaire) avec un RNN caractère-par-caractère
- **Exercice 2** : embeddings GloVe sur légendes Flickr8k (extraction + clustering + t-SNE)

---

## Structure du dépôt

```text
.
├─ scripts/                  # scripts des exercices (exo0..exo4)
├─ notebooks/                # notebook principal
├─ models/                   # modèles entraînés / export JSON (selon config)
├─ results_epoch_compare/    # résultats (textes générés, récapitulatif)
├─ data/                     # données (non versionnées si trop lourdes)
├─ TP51_RN_deep_debache.pdf  # rapport
├─ make_all.sh               # lancement automatique
├─ requirements.txt
>>>>>>> Update README and remove data/.gitignore
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
