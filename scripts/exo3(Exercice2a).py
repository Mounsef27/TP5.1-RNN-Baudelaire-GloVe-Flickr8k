#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import _pickle as pickle

DATA_DIR = "data"
filename = os.path.join(DATA_DIR, "flickr_8k_train_dataset.txt")
GLOVE_MODEL = os.path.join(DATA_DIR, "glove.6B.100d.txt")
outfile = os.path.join(DATA_DIR, "Caption_Embeddings.p")

def main():
    df = pd.read_csv(filename, delimiter="\t", header=None)
    nb_samples = df.shape[0]

    it = df.iterrows()
    allwords = []
    for _ in range(nb_samples):
        x = next(it)
        cap_words = str(x[1][1]).split()
        cap_wordsl = [w.lower() for w in cap_words]
        allwords.extend(cap_wordsl)

    unique = list(set(allwords))
    print("Nb mots uniques dans les légendes =", len(unique))

    listwords = []
    listembeddings = []
    cpt = 0

    fglove = open(GLOVE_MODEL, "r", encoding="utf-8")
    for line in fglove:
        row = line.strip().split()
        word = row[0]  # COMPLETE
        if (word in unique) or (word == "unk"):
            embedding = np.asarray(row[1:], dtype="float32")  # COMPLETE
            listwords.append(word)
            listembeddings.append(embedding)
            cpt += 1
            if cpt % 500 == 0 or word == "unk":
                print("word:", word, "embedded", cpt)
    fglove.close()

    nbwords = len(listembeddings)
    tembedding = len(listembeddings[0])
    print("Number of words =", nbwords, "Embedding size =", tembedding)

    embeddings = np.zeros((len(listembeddings)+2, tembedding+2), dtype="float32")
    for i in range(nbwords):
        embeddings[i, 0:tembedding] = listembeddings[i]

    # Ajout <start> et <end>
    listwords.append("<start>")
    embeddings[nbwords, tembedding] = 1.0

    listwords.append("<end>")
    embeddings[nbwords+1, tembedding+1] = 1.0

    print("Final embeddings shape:", embeddings.shape)
    print("Final words count:", len(listwords))

    with open(outfile, "wb") as pickle_f:
        pickle.dump([listwords, embeddings], pickle_f)

    print("Saved:", outfile)

if __name__ == "__main__":
    main()
