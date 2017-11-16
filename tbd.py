import config

import pandas as pd
import spotipy
import spotipy.util as util
from copy import deepcopy


def get_playlist_tracks(sp, username, playlist_id):
    playlist = sp.user_playlist_tracks(username,playlist_id)
    features = []
    ids = []
    while playlist['next']:
        playlist = sp.next(playlist)
        for i, item in enumerate(playlist['items']):
            # want song name, artist, id
            song_name = item['track']['name'] 
            first_artist = item['track']['artists'][0]['name'] 
            id = item['track']['id']
            ids.append(id)
            #feats = sp.audio_features('7FqsuPolV5GURA7p3HPQtn')
            # feat = sp.audio_features(str(id))[0]
            # feat.update({'song_name': song_name, 'first_artist': first_artist})
            # features.append(feat)#(song_name, first_artist, id, feats)
        break
    features = sp.audio_features(ids) 
    return features

# Spotify info
def get_basics_all_songs():
    scopes = "user-library-read user-top-read playlist-read-private playlist-read-collaborative"
    token = util.prompt_for_user_token(config.SPOTIFY_USERNAME, scopes,
                        client_id=config.CLIENT_ID, 
                        client_secret=config.CLIENT_SECRET, 
                        redirect_uri='http://localhost:8888/callback')

    sp = spotipy.Spotify(auth=token)
    songs = get_playlist_tracks(sp, config.SPOTIFY_USERNAME, '5Kzj5lmBPdDI1oPpOFzZzQ') 
    all_songs = pd.DataFrame(songs, columns=['Song_Name', 'Artist', 'ID'])
    #print all_songs
    #all_songs.to_pickle('all_songs.pickle')

def get_attributes_all_songs(path_to_pickle):
    df = pd.read_pickle(path_to_pickle)



get_basics_all_songs()





# playlist = sp.user_playlist("123640263", "0PGgtGa09RGgdTJHNAcZAE", 'tracks, next') 
# songs = playlist["tracks"]["items"] 
# ids = []
# for i in range(len(songs)): 
# 	ids.append(songs[i]["track"]["id"]) 
# features = sp.audio_features(ids) 
# df = pd.DataFrame(features)
# for song in features:
# 	name = [ playlist['tracks']['items'][x]['track']['name'] \
# 	for x in xrange(len(playlist['tracks']['items'])) \
# 		if playlist['tracks']['items'][x]['track']['id'] == song['id']]
# 	song['name'] = name[0]
# df.to_pickle('./dislikes')