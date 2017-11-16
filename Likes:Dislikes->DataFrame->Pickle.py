import pandas as pd 
import spotipy 
import pickle
sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 
cid ="6c8edf7ef6344e84a6dd6bd654d43cbc" 
secret = "94b58545a6fc43f4aa7a6632e3cfe9ff" 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 
playlist = sp.user_playlist("123640263", "6fPW6bbkEqCNiaVKLMDlep", 'tracks, next') 
songs = playlist["tracks"]["items"] 
ids = []
for i in range(len(songs)): 
	ids.append(songs[i]["track"]["id"]) 
features = sp.audio_features(ids) 
print features
df = pd.DataFrame(features + [songs[x]['track']['name'] for x in xrange(len(songs))])
df.to_pickle('.')

playlist = sp.user_playlist("123640263", "0PGgtGa09RGgdTJHNAcZAE", 'tracks, next') 
songs = playlist["tracks"]["items"] 
ids = []
for i in range(len(songs)): 
	ids.append(songs[i]["track"]["id"]) 
features = sp.audio_features(ids) 
print features
df = pd.DataFrame(features + [songs[x]['track']['name'] for x in xrange(len(songs))])
df.to_pickle('.')

# with open('songs.csv', 'w') as f:
# 	df.to_csv(path_or_buf=f)