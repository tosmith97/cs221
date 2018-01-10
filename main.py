# main
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib
import time, random
from get_song_data import get_playlist

# def leave():
# 	print 'Invalid input...quitting'
# 	quit()

def get_emotion(clf):
	in_str = raw_input('Describe your feelings, and press enter when you are finished: ')
	while in_str == '':
		in_str = raw_input('Describe your feelings, and press enter when you are finished: ')
	return clf.predict([in_str])[0]
	# probs = clf.predict_proba([in_str])
	# emotions_to_probs = {'excited':0, 'happy':1, 'relaxed':2, 'sad':3}
	# probs_to_emotions = {0:'excited', 1:'happy', 2:'relaxed', 3:'sad'}
	# correct = raw_input('Sounds like you feel %s. Does that sound right? (y/n)' % emotion)
	# if 'y' in correct or 'Y' in correct:
	# 	return emotion
	# elif 'n' in correct or 'N' in correct:
	# 	del probs[emotions_to_probs[emotion]]
	# 	emotion = probs_to_emotions[probs.index(max(probs))]
	# 	correct = raw_input('Got it. Maybe it\'s more like %s. Does that sound better? (y/n)' % emotion)
	# 	if 'y' in correct or 'Y' in correct:
	# 		return emotion
	# 	elif 'n' in correct or 'N' in correct:
	# 		del probs[emotions_to_probs[emotion]]
	# 		emotion = probs_to_emotions[probs.index(max(probs))]
	# 		correct = raw_input('I want to get this right. Are you feeling %s? (y/n)' % emotion)
	# 		if 'y' in correct or 'Y' in correct:
	# 			return emotion
	# 		elif 'n' in correct or 'N' in correct:
	# 			del probs[emotions_to_probs[emotion]]
	# 			emotion = probs_to_emotions[probs.index(max(probs))]
	# 			print "Got it. It seems you're feeling %s. Duly noted!" % emotion
	# 			return emotion
	# 		else:
	# 			leave()
	# 	else:
	# 		leave()
	# else:
	# 	leave()



def get_user():
	t = 10
	raw_input("Please enter your Spotify username. ")
	i = random.randint(0, t/2)
	time.sleep(i)
	print 'Training on your tastes'
	time.sleep(t-i)

def main():
	# call pickle model on input
	clf = joblib.load('models/svm_sentiment_clf.pkl')
	emotion = get_emotion(clf)
	# if emotion == 'relaxed':
	# 	emotion = 'chill'
	#get_user()
	get_playlist(emotion)


main()

# [excited, happy, relaxed, sad]