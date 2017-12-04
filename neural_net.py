import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM

from util import get_lh_data, get_emotion_data

def create_model(num_neurons=100):
    print('Build model...')
    model = Sequential()
    # model.add(Dense(256, activation='relu', input_dim=14))
    # model.add(Dropout(0.5))
    model.add(Dense(num_neurons, activation='relu', input_dim=14))
    #('Test score:', 0.4748)
    #('Test accuracy:', 0.78)

    #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_dim=13))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    return model


def train_lh_nn_model(num_neurons=100):
    batch_size = 32

    x, y = get_lh_data()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
    cvscores = []
    for train, test in kfold.split(x, y.flatten()):
        model = create_model(num_neurons)
        model.fit(x[train], y[train], epochs=100, batch_size=batch_size)

        scores = model.evaluate(x[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return np.mean(cvscores)
    #model.save('hate_love_nn.model')
    # print('Test score:', score)
    # print('Test accuracy:', acc)


def train_emot_nn_model(num_neurons=100):
    batch_size = 32

    x, y = get_emotion_data()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    cvscores = []
    for train, test in kfold.split(x, y.flatten()):
        model = create_model(num_neurons)
        model.fit(x[train], y[train], epochs=1, batch_size=batch_size)
        
        scores = model.evaluate(x[test], y[test], verbose=0)
        print scores
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return np.mean(cvscores)
    #model.save('hate_love_nn.model')
    # print('Test score:', score)
    # print('Test accuracy:', acc)


np.random.seed(11)
# train_nn_model()

# neuron_vals = [66, 44, 33, 26, 22, 19, 16, 14, 13]
# best_acc = -1
# best_val = -1
# for v in neuron_vals:
#     acc_for_model = train_emot_nn_model(v)
#     if acc_for_model > best_acc:
#         best_acc = acc_for_model
#         best_val = v
    
# print best_acc, best_val

train_lh_nn_model(72)

# neuron_vals = [109, 72, 54, 43, 36, 31, 27, 24, 21]
# best for lh: 72 neurons 

# neuron_vals = [66, 44, 33, 26, 22, 19, 16, 14, 13]
