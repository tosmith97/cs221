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
    playlist = sp.user_playlist('22gfmzpzyjgmxov5usnqrzkci', '5Kzj5lmBPdDI1oPpOFzZzQ', 'tracks, next')
    print playlist['tracks']
    songs = playlist['tracks']['items']
    print songs

get_all_songs()