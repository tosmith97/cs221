import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


numpy.random.seed(11)

sentiment_data = pd.read_csv('data/sentiment_data.csv', encoding='ISO-8859-1')
sentiment_data = shuffle(sentiment_data)
x = sentiment_data.drop('Sentiment', axis=1)
y = sentiment_data[['Sentiment']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.Text)
tfidf_transformer = TfidfTransformer()
X_tr = tfidf_transformer.fit_transform(X_train_counts)

test_counts = count_vect.fit_transform(X_test.Text)
X_te = tfidf_transformer.fit_transform(X_train_counts)


# truncate and pad input sequences
max_review_length = 50
X_train = sequence.pad_sequences(X_tr, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_te, maxlen=max_review_length)

# create the model
embedding_vecor_length = 16
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))