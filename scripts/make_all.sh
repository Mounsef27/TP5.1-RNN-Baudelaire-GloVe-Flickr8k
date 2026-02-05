#!/usr/bin/env bash
set -euo pipefail

echo "=== TP 5.1 - RNN / GloVe ==="

echo "[1/5] exo0.py (dataset Baudelaire)"
python3 exo0.py

echo "[2/5] exo1.py (train RNN)"
python3 exo1.py

echo "[3/5] exo2.py (generate text)"
python3 exo2.py

echo "[4/5] exo3.py (build Caption_Embeddings.p)"
python3 exo3.py

echo "[5/5] exo4.py (KMeans + t-SNE figure)"
python3 exo4.py

echo "=== DONE ==="
