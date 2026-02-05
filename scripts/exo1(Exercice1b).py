#!/usr/bin/env python3
import _pickle as pickle
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

SEQLEN = 10
outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
MODEL_NAME = "Baudelaire_RNN_len_" + str(SEQLEN)

def saveModel(model, savename):
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model", savename + ".yaml saved to disk")
    model.save_weights(savename + ".h5")
    print("Weights", savename + ".h5 saved to disk")

def main():
    [index2char, X_train, y_train, X_test, y_test] = pickle.load(open(outfile, "rb"))
    nb_chars = len(index2char)
    print("nb_chars =", nb_chars)

    model = Sequential()

    HSIZE = 128
    model.add(SimpleRNN(HSIZE, return_sequences=False,
                        input_shape=(SEQLEN, nb_chars), unroll=True))

    # ADD FULLY CONNECTED LAYER (output size ? = nb_chars)
    model.add(Dense(nb_chars))

    # ADD SOFTMAX
    model.add(Activation("softmax"))

    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    learning_rate = 0.001

    # CREATE OPTIMIZER & COMPILE
    opt = RMSprop(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    # FIT MODEL TO DATA
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(X_test, y_test),
              verbose=2)

    # EVALUATE TRAINED MODEL
    scores_train = model.evaluate(X_train, y_train, verbose=1)
    scores_test = model.evaluate(X_test, y_test, verbose=1)
    print("PERFS TRAIN: %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
    print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

    saveModel(model, MODEL_NAME)

if __name__ == "__main__":
    main()
