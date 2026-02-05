#!/usr/bin/env python3
import os
import numpy as np
import _pickle as pickle

DATA_DIR = "data"
INFILE = os.path.join(DATA_DIR, "fleurs_mal.txt")

SEQLEN = 10
STEP = 1
ratio_train = 0.8

def main():
    bStart = False
    with open(INFILE, "r", encoding="utf8") as fin:
        lines = fin.readlines()

    lines2 = []
    for line in lines:
        line = line.strip().lower()
        if ("charles baudelaire avait un ami" in line and bStart == False):
            print("START")
            bStart = True
        if ("end of the project gutenberg ebook of les fleurs du mal, by charles baudelaire" in line):
            print("END")
            break
        if (bStart == False or len(line) == 0):
            continue
        lines2.append(line)

    text = " ".join(lines2)

    # chars = dictionnaire des symboles (caractères) présents dans le texte
    chars = sorted(set([c for c in text]))
    nb_chars = len(chars)

    print("Text length:", len(text))
    print("nb_chars:", nb_chars)

    # Génération des séquences d'entrée et labels (caractère suivant)
    input_chars = []
    label_chars = []
    for i in range(0, len(text) - SEQLEN, STEP):
        # Append input of size SEQLEN
        input_chars.append(text[i:i+SEQLEN])
        # Append output (label) of size 1
        label_chars.append(text[i+SEQLEN])

    nbex = len(input_chars)
    print("nbex =", nbex)

    # mapping char -> index
    char2index = dict((c, i) for i, c in enumerate(chars))
    index2char = dict((i, c) for i, c in enumerate(chars))

    # One-hot encoding
    X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=bool)
    y = np.zeros((len(input_chars), nb_chars), dtype=bool)

    for i, input_char in enumerate(input_chars):
        for j, ch in enumerate(input_char):
            # Fill X at correct index
            X[i, j, char2index[ch]] = 1
        # Fill y at correct index
        y[i, char2index[label_chars[i]]] = 1

    # Split train/test
    nb_train = int(round(len(input_chars) * ratio_train))
    print("nb tot=", len(input_chars), "nb_train=", nb_train)

    X_train = X[0:nb_train, :, :]
    y_train = y[0:nb_train, :]

    X_test = X[nb_train:, :, :]
    y_test = y[nb_train:, :]

    print("X train.shape=", X_train.shape)
    print("y train.shape=", y_train.shape)
    print("X test.shape=", X_test.shape)
    print("y test.shape=", y_test.shape)

    outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
    with open(outfile, "wb") as pickle_f:
        pickle.dump([index2char, X_train, y_train, X_test, y_test], pickle_f)

    print("Saved:", outfile)

if __name__ == "__main__":
    main()
