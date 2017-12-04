import pickle
import pandas as pd
import numpy as np
from util import get_emotion_data

from sklearn import ensemble
from sklearn.base import TransformerMixin
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


def train_sentiment_clfs():
    sentiment_data = pd.read_csv('data/sentiment_data.csv', encoding='ISO-8859-1')
    sentiment_data = shuffle(sentiment_data)
    x = sentiment_data.drop('Sentiment', axis=1)
    y = sentiment_data[['Sentiment']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

    sentiment_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='log', penalty='l2',
                                            alpha=1e-3, random_state=42)),
    ])


    sentiment_clf_svm.fit(X_train.Text, y_train)
    svm_predicted = sentiment_clf_svm.predict(X_test)

    sentiment_clf_nb = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()),
    ])

    sentiment_clf_nb.fit(X_train.Text, y_train)
    nb_predicted = sentiment_clf_nb.predict(X_test)

    print 'SVM: ', np.mean(svm_predicted == y_test) 
    print 'NB: ', np.mean(nb_predicted == y_test) 

    # joblib.dump(sentiment_clf_svm, 'models/svm_sentiment_clf.pkl') 
    # joblib.dump(sentiment_clf_nb, 'models/nb_sentiment_clf.pkl') 


def train_gs_classifier():
    sentiment_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42)),
    ])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'tfidf__use_idf': (True, False),
                'clf__alpha': (1e-2, 1e-3),
                'tfidf__norm': ('l1', 'l2', None),
    }
    gs_clf = GridSearchCV(sentiment_clf_svm, parameters, n_jobs=-1)

    sentiment_data = pd.read_csv('data/sentiment_data.csv', encoding='ISO-8859-1')
    sentiment_data = shuffle(sentiment_data)
    x = sentiment_data.drop('Sentiment', axis=1)
    y = sentiment_data[['Sentiment']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

    gs_clf.fit(x.Text, y.values.ravel())

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    print gs_clf.best_score_     
    joblib.dump(gs_clf.best_estimator_, 'models/svm_sentiment_clf.pkl') 

np.random.seed(11)
train_gs_classifier()
#train_sentiment_clfs()