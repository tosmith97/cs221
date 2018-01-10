import pickle
import pandas as pd
import numpy as np
from util import get_emotion_data

from sklearn import ensemble
from sklearn.base import TransformerMixin
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier  
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def train_emotion_classifier():
    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)
    # parameters = dict(gamma=gamma_range, C=C_range)
    # print parameters

    # gs_clf = GridSearchCV(sentiment_clf_svm, parameters, n_jobs=-1)

    emotion_data = pd.read_csv('data/emotion_data.csv')
    x = emotion_data.drop('Emotion', axis=1)
    y = emotion_data[['Emotion']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

    # gs_clf.fit(x, y.values.ravel())

    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # print gs_clf.best_score_     

    parameters = {
    "estimator__C": [1,2,4,8],
    "estimator__kernel": ["poly","rbf", "linear"],
    "estimator__degree":[1, 2, 3, 4],
}
    classifier = OneVsRestClassifier((SVC()))
    grid = GridSearchCV(classifier, parameters)
    grid.fit(x, y)

    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    # joblib.dump(gs_clf.best_estimator_, 'models/svm_sentiment_clf.pkl') 


def train_taste_classifier():
    taste_data = pd.read_csv('data/taste_data.csv')
    x = taste_data.drop('Love', axis=1)
    y = taste_data[['Love']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(x, y)

    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    # joblib.dump(gs_clf.best_estimator_, 'models/svm_sentiment_clf.pkl') 


np.random.seed(11)
train_emotion_classifier()
#train_taste_classifier()
