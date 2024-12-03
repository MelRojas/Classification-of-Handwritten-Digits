import numpy as np
import sklearn
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
import pandas as pd
from operator import itemgetter

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    # target_pred = model.predict_proba(features_test)
    target_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, target_pred)
    # print(f'best estimator: {model}\naccuracy: {score}\n')
    # return score
    return f'best estimator: {model}\naccuracy: {score}\n'


# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28)
y_train = y_train

# Get unique labels in the training and test sets
unique_X_train_labels = np.unique(x_train)
unique_y_train_labels = np.unique(y_train)

X_train, X_test, y_train, y_test = train_test_split(x_train[:6000], y_train[:6000], test_size=0.3, random_state=40)
train_class_proportions = pd.Series(y_train).value_counts(normalize=True).sort_index()

normalizer = Normalizer()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.fit_transform(X_test)

kn_grid_params = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
grid_clf_kn = GridSearchCV(KNeighborsClassifier(),
                        kn_grid_params,
                        scoring='accuracy',
                        n_jobs=-1)
grid_clf_kn.fit(X_train_norm, y_train)
kn_grid_params = grid_clf_kn.best_params_

print('K-nearest neighbours algorithm\n', fit_predict_eval(KNeighborsClassifier(n_neighbors=kn_grid_params['n_neighbors'],
                                    weights=kn_grid_params['weights'],
                                    algorithm=kn_grid_params['algorithm']),
                 X_train_norm, X_test_norm, y_train, y_test))


rf_grid_params = {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'], 'class_weight': ['balanced', 'balanced_subsample']}
grid_clf_rf = GridSearchCV(RandomForestClassifier(random_state=40),
                           rf_grid_params,
                           scoring='accuracy',
                           n_jobs=-1)
grid_clf_rf.fit(X_train_norm, y_train)
rf_grid_params = grid_clf_rf.best_params_

print('Random forest algorithm\n', fit_predict_eval(RandomForestClassifier(n_estimators=rf_grid_params['n_estimators'],
                                        max_features=rf_grid_params['max_features'],
                                        class_weight=rf_grid_params['class_weight'],
                                        random_state=40),
                 X_train_norm, X_test_norm, y_train, y_test))



