from nltk.corpus import wordnet as wn
import pandas as pd
from collections import defaultdict
import pickle

happy = ['happy', 'content', 'cheery', 'untroubled', 'delighted', 'pleased', 'glad', 'jolly', 'cheerful', 'merry', 'ecstatic', 'upbeat', 'lively']
sad = ['sad', 'unhappy', 'sorrowful', 'miserable', 'awful', 'melancholy', 'mournful', 'dismal']
excited = ['excited', 'thrilled', 'electrified', 'aroused', 'stimulated', 'animated', 'delighted']
relaxed = ['relaxed', 'serene', 'casual', 'laid-back', 'calm', 'tranquil', 'satisfied', 'cozy']

def get_emotion_sentences(words, emotion_sentences):
    synsets = defaultdict(list)
    key_word = words[0]
    for w in words:
        synsets[w] = wn.synsets(w)

    for emot, syn in synsets.iteritems():
        for s in syn:
            examples = wn.synset(s._name).examples()
            for e in examples:
                emotion_sentences[key_word].append(e)
            
            hypernyms = wn.synset(s._name).hypernyms()  
            for h in hypernyms:
                h_examples = wn.synset(h._name).examples()
                for e in h_examples:
                    emotion_sentences[key_word].append(e)


def main():
    emotion_sentences = defaultdict(list)
    emotions = [happy, sad, excited, relaxed]
    dfs = []

    for e in emotions:
        get_emotion_sentences(e, emotion_sentences)

    with open('data/sentiment_tagged_sentences.txt', 'w') as f:
        for e in emotion_sentences:
            for s in emotion_sentences[e]:
                f.write('%s,%s\n' % (e,s))

# main()
df = pd.read_csv('data/sentiment_tagged_sentences.txt', sep=",", header=None)
df.columns = ['Sentiment', 'Text']
df.to_csv('data/sentiment_tagged_sentences.csv', index=False)