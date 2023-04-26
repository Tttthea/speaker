#preprocessing
#classification
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

import xgboost as xgb
from sklearn import metrics
from sklearn.svm import SVC

def generate():
    """generate dataset to train"""
    train = np.array(pd.read_csv('train.csv', index_col=0))
    X = train[:, :-1]
    y = train[:, -1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return scaler, X_train, X_test, y_train, y_test

def gen_cross():
    rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
    train = np.array(pd.read_csv('train.csv', index_col=0))
    X = train[:, :-1]
    y = train[:, -1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    for train_index, test_index in rkf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return scaler, X_train, X_test, y_train, y_test
def train():
    """
    Voting on XGBoost, Logistic Regression, SVM classifiers, which shown the best performances in training set
    train-test-split 7/3
    """
    # _, X_train, X_test, y_train, y_test = generate()
    _, X_train, X_test, y_train, y_test = gen_cross()
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05).fit(X_train, y_train)

    lr = LogisticRegression(random_state=0).fit(X_train, y_train)
    svc = SVC().fit(X_train, y_train)
    vot = VotingClassifier(estimators=[('xgboost', gbm), ('svc', svc), ('lr', lr)], voting='hard').fit(X_train, y_train)
    y_pred = vot.predict(X_test)
    print(f'accuracy on testing set: {metrics.accuracy_score(y_test, y_pred)}')
    print(f'classification performance: \n {classification_report(y_test, y_pred)}')
    return vot


def predict(data):
    """predict gender on given audio"""
    model = train()
    gender = model.predict(data)
    return gender


