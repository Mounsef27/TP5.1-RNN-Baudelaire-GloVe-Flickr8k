#!/usr/bin/env python3
import numpy as np
import _pickle as pickle
from keras.models import model_from_yaml

SEQLEN = 10
outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
nameModel = "Baudelaire_RNN_len_" + str(SEQLEN)

def loadModel(savename):
    with open(savename + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model", savename + ".yaml loaded")
    model.load_weights(savename + ".h5")
    print("Weights", savename + ".h5 loaded")
    return model

def sampling(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    predsN = np.power(preds, 1.0/temperature)
    predsN /= np.sum(predsN)
    probas = np.random.multinomial(1, predsN, 1)
    return np.argmax(probas)

def main():
    [index2char, X_train, y_train, X_test, y_test] = pickle.load(open(outfile, "rb"))

    model = loadModel(nameModel)
    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
    model.summary()

    nb_chars = len(index2char)

    seed = 15608 % X_train.shape[0]
    char_init = ""
    for i in range(SEQLEN):
        char = index2char[np.argmax(X_train[seed, i, :])]
        char_init += char
    print("CHAR INIT:", char_init)

    test = np.zeros((1, SEQLEN, nb_chars), dtype=bool)
    test[0, :, :] = X_train[seed, :, :]

    nbgen = 400
    gen_char = char_init
    temperature = 0.5

    for _ in range(nbgen):
        preds = model.predict(test, verbose=0)[0]  # shape (nb_chars,)
        next_ind = sampling(preds, temperature=temperature)  # SAMPLING
        next_char = index2char[next_ind]  # CONVERT INDEX -> CHAR
        gen_char += next_char

        for i in range(SEQLEN-1):
            test[0, i, :] = test[0, i+1, :]
        test[0, SEQLEN-1, :] = 0
        test[0, SEQLEN-1, next_ind] = 1

    print("\nGenerated text:\n")
    print(gen_char)

if __name__ == "__main__":
    main()
