import config

import pandas as pd
import spotipy
import spotipy.util as util
from copy import deepcopy


def setup_spotipy():
    scopes = "user-library-read user-top-read playlist-read-private playlist-read-collaborative"
    token = util.prompt_for_user_token(config.SPOTIFY_USERNAME, scopes,
                        client_id=config.CLIENT_ID, 
                        client_secret=config.CLIENT_SECRET, 
                        redirect_uri='http://localhost:8888/callback')

    sp = spotipy.Spotify(auth=token)
    return sp   


def get_feats_for_song_in_playlist(sp, spotify_username, playlist_id, outfile_path):
    playlist = sp.user_playlist_tracks(spotify_username, playlist_id)
    features = []
    ids = []
    while playlist['next']:
        playlist = sp.next(playlist)
        for i, item in enumerate(playlist['items']):
            # song_name = item['track']['name'] 
            # first_artist = item['track']['artists'][0]['name'] 
            song_ids = item['track']['id']
            ids.append(song_ids)
    
    features = sp.audio_features(ids) 
    df = pd.DataFrame(features)
    df.to_pickle(outfile_path)


sp = setup_spotipy()

# for all songs
get_feats_for_song_in_playlist(sp, config.SPOTIFY_USERNAME, config.ALL_SONGS_PLAYLIST_ID, 'all_song_feats.pickle')

