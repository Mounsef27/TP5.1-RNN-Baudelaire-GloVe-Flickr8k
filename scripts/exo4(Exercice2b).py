#!/usr/bin/env python3
import os
import numpy as np
import _pickle as pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATA_DIR = "data"
outfile = os.path.join(DATA_DIR, "Caption_Embeddings.p")
FIG_OUT = os.path.join(DATA_DIR, "tsne_flickr8k.png")

def main():
    [listwords, embeddings] = pickle.load(open(outfile, "rb"))
    print("embeddings:", embeddings.shape)

    tembedding = 100
    E = embeddings.copy().astype("float32")

    # L2 NORMALIZATION (only glove part 0..99)
    for i in range(E.shape[0]):
        v = E[i, 0:tembedding]
        n = np.linalg.norm(v)
        if n > 0:
            E[i, 0:tembedding] = v / n

    # KMeans
    kmeans = KMeans(n_clusters=10, init="random", max_iter=1000, n_init=10, random_state=0)
    kmeans.fit(E)  # COMPLETE
    clustersID = kmeans.labels_
    clusters = kmeans.cluster_centers_

    pointsclusters = np.zeros((10, 2), dtype="float32")  # INIT
    indclusters = np.zeros((10, 21), dtype=int)          # INIT

    for i in range(10):
        norm = np.linalg.norm((clusters[i] - E), axis=1)
        inorms = np.argsort(norm)
        indclusters[i, :] = inorms[:21]

        print("\nCluster", i, "=", listwords[indclusters[i, 0]])
        for j in range(1, 21):
            print("  mot:", listwords[indclusters[i, j]])

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, verbose=2, init="pca", early_exaggeration=24, random_state=0)
    points2D = tsne.fit_transform(E)

    for i in range(10):
        pointsclusters[i, :] = points2D[int(indclusters[i, 0])]

    cmap = cm.tab10
    plt.figure(figsize=(3.841, 7.195), dpi=200)
    plt.set_cmap(cmap)
    plt.scatter(points2D[:, 0], points2D[:, 1], c=clustersID, s=3, edgecolors="none", cmap=cmap, alpha=1.0)
    plt.scatter(pointsclusters[:, 0], pointsclusters[:, 1], c=range(10), marker="+", s=600, edgecolors="none", cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=range(10))
    plt.title("t-SNE GloVe Flickr8k + centres KMeans")
    plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    print("Saved figure:", FIG_OUT)

if __name__ == "__main__":
    main()
