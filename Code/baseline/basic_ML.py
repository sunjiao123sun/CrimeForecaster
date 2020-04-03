import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import itertools as it
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics_sk
from sklearn.preprocessing import MinMaxScaler

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report
import pickle
import statistics
import os
import argparse

ml_model = 'LR'
city = 'CHICAGO'
month = 8



def sigmoid(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            # print("a :", array[i][j])
            array[i][j] = 1/(1 + np.exp(-array[i][j]))
            # print("b :", array[i][j])
    return array



def evaluate_micro_plot(Y_test, y_score, n_classes):
    '''
    Y_test: the label for test data;
    y_score: the predict result for rest data;
    '''
    Y_test = np.array(Y_test)
    y_score = np.array(y_score)
    Y_test = Y_test.reshape(-1,8)
    y_score = y_score.reshape(-1,8)
    print("non-zero elements in prediction is {} and in truth is {} ".format(np.count_nonzero(
        y_score), np.count_nonzero(Y_test)))
    # For each class
    macro_f1 = metrics_sk.f1_score(Y_test, y_score, average = 'macro')
    micro_f1 = metrics_sk.f1_score(Y_test, y_score, average = 'micro')
    print(classification_report(Y_test, y_score))
    print("The average macro-F1 score is {}, average micro-F1 score is {}".format(macro_f1, micro_f1))
    return



train = np.load('../CrimeForecaster/CRIME-CHICAGO/8/train.npz'%(city,month), allow_pickle = True)
test = np.load('../CrimeForecaster/CRIME-CHICAGO/8/test.npz'%(city,month), allow_pickle=True)
val = np.load('../CrimeForecaster/CRIME-CHICAGO/8/val.npz'%(city,month),allow_pickle=True)

features_train, features_test, labels_train, labels_test = train["x"], test["x"], train["y"], test["y"]

features_train_reshape = features_train.reshape(features_train.shape[0],-1)
features_test_reshape = features_test.reshape(features_test.shape[0],-1)

labels_train_reshape = labels_train.reshape(labels_train.shape[0],-1)
labels_test_reshape = labels_test.reshape(labels_test.shape[0],-1)



print("features_train_shape:",features_train_reshape.shape)
print("labels_train_shape:",labels_train_reshape.shape)
print("features_test_shape:",features_test_reshape.shape)
print("labels_test_shape:",labels_test_reshape.shape)


#val_x, val_y = val["x"], val["y"]
#val_x, val_y = val_x.reshape((val_x.shape[0],
#                                      num_timeslots, -1)), \
#               np.squeeze(val_y, axis = 1)


def baseline_model(X_train, y_train, X_test, y_test, model):
    if model == 'RF':
        print("MODEL: RANDOM FOREST")
        clf_rf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=10)
        clf_rf.fit(X_train, y_train)
        y_predict = clf_rf.predict(X_test)
        evaluate_micro_plot(y_test, y_predict, 2)
        return y_predict, y_test

    elif model == 'SVR':
        print("MODEL: SVR")
        clf_svr = OneVsRestClassifier(SVC(kernel='rbf',class_weight='balanced',gamma='auto'))
        clf_svr.fit(X_train, y_train)
        y_predict = clf_svr.predict(X_test)
        Y_test = np.array(y_test)
        y_score = np.array(y_predict)
        print("non-zero elements in prediction is {} and in truth is {} ".format(np.count_nonzero(
            y_score), np.count_nonzero(Y_test)))
        # For each class
        macro_f1 = metrics_sk.f1_score(Y_test, y_score, average='macro')
        micro_f1 = metrics_sk.f1_score(Y_test, y_score, average='micro')
        print("The average macro-F1 score is {}, average micro-F1 score is {}".format(macro_f1, micro_f1))
        return y_score, Y_test
    elif model == 'DecisionTree':
        print("MODEL: DECISION TREE")
        clf_dt = DecisionTreeClassifier(random_state=0)
        clf_dt.fit(X_train, y_train)
        y_predict_dt = clf_dt.predict(X_test)

        average_precision_dtc = evaluate_micro_plot(y_test, y_predict_dt, 2)
        return y_predict_dt, y_test
    elif model == 'LR':
        print("MODEL: LOGISTIC REGRESSION")
        clf_lr = OneVsRestClassifier(LogisticRegression())
        clf_lr.fit(X_train, y_train)
        y_predict_lr = clf_lr.predict(X_test)
        #ss = MinMaxScaler(feature_range=(0, 1))
        pre_ratio = np.count_nonzero(y_test) / np.size(y_test)  # the ratio of normal events in the test set
        print("The ratio of abnormal events in the test set is:", pre_ratio)
        average_precision_lr = evaluate_micro_plot(y_test, y_predict_lr, 2)
        return y_predict_lr, y_test
    elif model == 'KNN':
        print("MODEL: KNN")
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        y_predict_neigh = neigh.predict(X_test)
        average_precision_neigh = evaluate_micro_plot(y_test, y_predict_neigh, 2)
        return y_predict_neigh, y_test
    elif model == 'extra_tree':
        print("MODEL: EXTRA TREE")
        clf_et = ExtraTreesClassifier(random_state=0,max_depth=3)
        clf_et.fit(X_train, y_train)
        y_predict_et = clf_et.predict(X_test)
        average_precision_etc = evaluate_micro_plot(y_test, y_predict_et, 2)
        print(average_precision_etc)
        return y_predict_et, y_test

print(np.shape(features_train_reshape), np.shape(labels_train_reshape), np.shape(features_test_reshape), np.shape(labels_test_reshape))
y_pred, y_test = baseline_model(features_train_reshape, labels_train_reshape, features_test_reshape, labels_test_reshape, ml_model)

pickle.dump(y_test, open('./result/%s/%s/labels_test%s%s%d.pkl'%(city,ml_model,ml_model,city,month), 'wb'))
pickle.dump(y_pred, open('./result/%s/%s/predict%s%s%d.pkl'%(city,ml_model,ml_model,city,month), 'wb'))

