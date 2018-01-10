import pandas as pd
import numpy as np
import pickle
import itertools

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


def get_feats_df(pickle):
    df = pd.read_pickle(pickle)
    df = df[['acousticness','instrumentalness', 'key', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
    return df


def get_lh_data():
    love_train = get_feats_df('data/song_features/love_feats.pickle')
    love_train['Love'] = 1

    hate_train = get_feats_df('data/song_features/hate_feats.pickle')
    hate_train['Love'] = 0

    tot_train_df = pd.concat([love_train, hate_train])
    tot_train_df = shuffle(tot_train_df)

    x = tot_train_df.drop('Love', axis=1)
    y = tot_train_df[['Love']]
    return x.values, y.values


def get_emotion_data():
    happy_df = get_feats_df('data/song_features/happy_feats.pickle')
    happy_df['Emotion'] = 0

    sad_df = get_feats_df('data/song_features/sad_feats.pickle')
    sad_df['Emotion'] = 1

    excited_df = get_feats_df('data/song_features/excited_feats.pickle')
    excited_df['Emotion'] = 2

    chill_df = get_feats_df('data/song_features/chill_feats.pickle')
    chill_df['Emotion'] = 3

    tot_train_df = pd.concat([happy_df, sad_df, excited_df, chill_df])
    tot_train_df = shuffle(tot_train_df)

    x = tot_train_df.drop('Emotion', axis=1)
    y = tot_train_df[['Emotion']]
    return x.values, y.values

def get_sentiment_data():
    sentiment_data = pd.read_csv('data/sentiment_data.csv', encoding='ISO-8859-1')
    sentiment_data = shuffle(sentiment_data)
    x = sentiment_data.drop('Sentiment', axis=1)
    y = sentiment_data[['Sentiment']]

    return x, y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_confusion_matrix(y_test, y_pred, class_names, plt_title):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title=plt_title)

    # Plot normalized confusion matrix
    plt.figure()
    norm_title = 'Normalized ' + plt_title
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title=norm_title)

    plt.show()