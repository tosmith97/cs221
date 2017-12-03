from nltk.corpus import wordnet as wn
from collections import defaultdict
import pickle

emotions = ['happy', 'content', 'cheery', 'untroubled', 'delighted', 'pleased', 
            'sad', 'unhappy', 'sorrowful', 'miserable', 'awful', 'melancholy' 
            'excited', 'thrilled', 'electrified', 'aroused', 'stimulated'
            'relaxed']

synsets = defaultdict(list)
emotion_sentences = defaultdict(list)

for e in emotions:
    synsets[e] = wn.synsets(e)

for emot, syn in synsets.iteritems():
    for s in syn:
        examples = wn.synset(s._name).examples()
        for e in examples:
            emotion_sentences[emot].append(e)
        
        hypernyms = wn.synset(s._name).hypernyms()  
        for h in hypernyms:
            h_examples = wn.synset(h._name).examples()
            for e in h_examples:
                emotion_sentences[emot].append(e)

tot = 0
for e in emotion_sentences:
    tot += len(emotion_sentences[e])
print tot    
with open('emotion_sentences_dict.pickle', 'wb') as f:
        pickle.dump(emotion_sentences, f)


# other strats:
# take input string, tokenize and remove stop words, then do comparison metric between every word and the 4 emotion words,assigning e word a weight

# hand-label more sentences then use regression, nn, etc