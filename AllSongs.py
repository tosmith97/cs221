import config

import pandas as pd
import spotipy
import spotipy.util as util


# Spotify info
def get_all_songs():
    scopes = "user-library-read user-top-read playlist-read-private playlist-read-collaborative"
    token = util.prompt_for_user_token(config.SPOTIFY_USERNAME, scopes,
                        client_id=config.CLIENT_ID, 
                        client_secret=config.CLIENT_SECRET, 
                        redirect_uri='http://localhost:8888/callback')

    sp = spotipy.Spotify(auth=token)
    songs = []
    playlist = sp.user_playlist('22gfmzpzyjgmxov5usnqrzkci', '5Kzj5lmBPdDI1oPpOFzZzQ', 'tracks, next')
    while playlist:
        songs += [playlist['tracks']['items'][i]['track']['name'] for i in xrange(len(playlist['tracks']['items']))]
        # for i in xrange(len(playlist['tracks']['items'])):
        #     print playlist['tracks']['items'][i]['track']['name']
        # songs = playlist['tracks']['items']
        playlist = sp.user_playlist('22gfmzpzyjgmxov5usnqrzkci', '5Kzj5lmBPdDI1oPpOFzZzQ')

get_all_songs()