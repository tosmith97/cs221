import config

import sys, pickle, spotipy, random
import spotipy.util as util
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from enum import Enum
from util import get_feats_df

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM

class Emotion(Enum):
    HAPPY = 0
    SAD = 1
    EXCITED = 2
    CHILL = 3


def setup_spotipy():
    scopes = "user-library-read user-top-read playlist-read-private playlist-read-collaborative"
    token = util.prompt_for_user_token(config.SPOTIFY_USERNAME, scopes,
                        client_id=config.CLIENT_ID, 
                        client_secret=config.CLIENT_SECRET, 
                        redirect_uri='http://localhost:8888/callback')

    sp = spotipy.Spotify(auth=token)
    return sp   


def get_feats_for_song_in_playlist(sp, spotify_username, playlist_id, outfile_path):
    print "outfile path: ", outfile_path
    playlist = sp.user_playlist_tracks(spotify_username, playlist_id)
    features = []
    counter = 0
    while playlist['next']:
        #batch_features = []
        batch_ids = []

        for i, item in enumerate(playlist['items']):
            if counter > 400:
                print counter 
                print item
                print
            song_name = item['track']['name'] 
            first_artist = item['track']['artists'][0]['name'] 
            song_id = item['track']['id']
            batch_ids.append(song_id)
            counter += 1
        
        playlist = sp.next(playlist)

        try:
            feats = sp.audio_features(batch_ids)
            features.extend(sp.audio_features(batch_ids))
        except AttributeError:
            continue
           
        if counter % 100 == 0:
            print 'on song ', counter
    
    print len(features)
    print counter
    #print features
    features = [f for f in features if f is not None]
    df = pd.DataFrame(features)
    df.to_pickle(outfile_path)
    print 'stored dataframe in pickle!'


def learn_love_hate_model(love_pickle, hate_pickle):
    love_train = get_feats_df(love_pickle)
    love_train['Love'] = 1

    hate_train = get_feats_df(hate_pickle)
    hate_train['Love'] = 0

    tot_train_df = pd.concat([love_train, hate_train])

    x = tot_train_df.drop('Love', axis=1)
    y = tot_train_df[['Love']]

    lr = LogisticRegression()
    lr.fit(x, y.values.ravel())
    with open('love_hate_model.pickle', 'wb') as f:
        pickle.dump(lr, f)
    print lr.score(x, y)


def learn_emotion_model(happy_pickle, sad_pickle, excited_pickle, chill_pickle):
    happy_df = get_feats_df(happy_pickle)
    happy_df['Emotion'] = 0

    sad_df = get_feats_df(sad_pickle)
    sad_df['Emotion'] = 1

    excited_df = get_feats_df(excited_pickle)
    excited_df['Emotion'] = 2

    chill_df = get_feats_df(chill_pickle)
    chill_df['Emotion'] = 3

    tot_train_df = pd.concat([happy_df, sad_df, excited_df, chill_df])

    x = tot_train_df.drop('Emotion', axis=1)
    y = tot_train_df[['Emotion']]

    lr = LogisticRegression()
    lr.fit(x, y.values.ravel())
    with open('emotion_model.pickle', 'wb') as f:
        pickle.dump(lr, f)
    print lr.score(x, y)


def get_songs_from_ids(sp, ids):
    all_songs = []
    for i in xrange(len(ids)):
        results = sp.track(ids[i])
        all_songs.append((results['artists'][0]['name'], results['name']))
    return all_songs


def get_songIDs_for_emotion(emotion):
    simple_all_songs_df = get_feats_df('song_features/all_song_feats.pickle')
    orig_all_songs = pd.read_pickle('song_features/all_song_feats.pickle')
    simple_all_songs_df = shuffle(simple_all_songs_df)
    orig_all_songs = shuffle(orig_all_songs)

    love_hate_model = pickle.load(open('love_hate_model.pickle', 'rb'))
    # NN - love_hate_model = load_model('hate_love_nn.model')
    
    emotion_model = pickle.load(open('emotion_model.pickle', 'rb'))
    NUM_SONGS = 5
    ids = []
    emotion_idx = Emotion[emotion].value
    for index, row in simple_all_songs_df.iterrows():
        # see if it's certain emotion
        emot_prob = emotion_model.predict_proba(row)

        # this might break in future
        emot_prob = emot_prob.reshape((-1,1))
        pred_idx = np.argmax(emot_prob) 
        if pred_idx == emotion_idx and max(emot_prob) > 0.5:
            # print emot_prob
            # see if we like it 
            #print love_hate_model.predict(np.reshape(row.values, (1 ,14)))
            if love_hate_model.predict(row) == 1:
                tempo = simple_all_songs_df.iloc[index]['tempo']
                acousticness = simple_all_songs_df.iloc[index]['acousticness']
                fitted_row = orig_all_songs.loc[(orig_all_songs['tempo'] == tempo) & (orig_all_songs['acousticness'] == acousticness)]
                song_id = fitted_row['id'].values[0]
                # song_id = orig_all_songs.iloc[fitted_row]['id']
                ids.append(song_id)

        if len(ids) >= NUM_SONGS:
            break
    return ids

def print_playlist(emotion, playlist):
    print 'You said you are feeling', emotion
    print 'Here is a list of songs for you to listen to!'
    for artist, song_name in playlist:
        print song_name, ' by ', artist

sp = setup_spotipy()

#train_rnn_models()

emotion = sys.argv[-1] # HAPPY, SAD, EXCITED, CHILL
ids = get_songIDs_for_emotion(emotion.upper())
playlist = get_songs_from_ids(sp, ids)
print_playlist(emotion, playlist)# for all songs


#get_feats_for_song_in_playlist(sp, config.SPOTIFY_USERNAME, config.ALL_SONGS_PLAYLIST_ID, 'all_song_feats.pickle')

# each emotion
#get_feats_for_song_in_playlist(sp, config.SPOTIFY_USERNAME, config.HAPPY_PLAYLIST_ID, 'happy_feats.pickle')
#get_feats_for_song_in_playlist(sp, config.SPOTIFY_USERNAME, config.SAD_PLAYLIST_ID, 'sad_feats.pickle')
#get_feats_for_song_in_playlist(sp, config.SPOTIFY_USERNAME, config.CHILL_PLAYLIST_ID, 'chill_feats.pickle')
#get_feats_for_song_in_playlist(sp, config.SPOTIFY_USERNAME, config.EXCITED_PLAYLIST_ID, 'excited_feats.pickle')
# get_feats_for_song_in_playlist(sp, '123640263', config.LIKE_PLAYLIST_ID, 'like_feats.pickle')
# get_feats_for_song_in_playlist(sp, '123640263', config.DISLIKE_PLAYLIST_ID, 'hate_feats.pickle')

# learn_love_hate_model('love_feats.pickle', 'hate_feats.pickle')
# learn_emotion_model('song_features/happy_feats.pickle', 'song_features/sad_feats.pickle', 'song_features/excited_feats.pickle', 'song_features/chill_feats.pickle')

