import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle


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