import numpy as np

from sklearn import ensemble
from util import get_emotion_data, get_lh_data, compute_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.externals import joblib

np.random.seed(11)

# x, y = get_emotion_data()
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
#           'learning_rate': 0.01, 'loss': 'deviance'}
# clf = ensemble.GradientBoostingClassifier(**params)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# acc = accuracy_score(y_test, predictions)
# print 'Emotion accuracy: ', acc
# compute_confusion_matrix(y_test, predictions, ['sad', 'happy', 'relaxed', 'excited'], 'Confusion Matrix for Song Emotion BRC')
# #print confusion_matrix(y_test, predictions, labels=['sad', 'happy', 'relaxed', 'excited'])
# # joblib.dump(clf, 'models/br_emotion_clf.pkl') 


x, y = get_lh_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

params = {'n_estimators': 5000, 'max_depth': 3, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'deviance'}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)
compute_confusion_matrix(y_test, predictions, ['Love', 'Hate'], 'Confusion Matrix for Song Taste BRC')
print 'Love/Hate accuracy:', acc
# joblib.dump(clf, 'models/br_loveHate_clf.pkl') 