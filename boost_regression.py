from sklearn import ensemble
from util import get_emotion_data, get_lh_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.externals import joblib


x, y = get_emotion_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'deviance'}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print 'Emotion accuracy: ', acc
joblib.dump(clf, 'models/br_emotion_clf.pkl') 


x, y = get_lh_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'deviance'}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print 'Love/Hate accuracy:', acc
joblib.dump(clf, 'models/br_loveHate_clf.pkl') 