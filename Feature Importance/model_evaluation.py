# =============================================================================
# HOMEWORK 6 - Model Evaluation
# First part: Random Forest Classification with Cross Validation Leave-One-Out on breast cancer data
# Second part: Friedman test on algo_performance dataset
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import friedmanchisquare
import numpy as np
from sklearn.metrics import confusion_matrix

# import dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# create leave-one-out cv procedure
cv = LeaveOneOut()
# enumerate splits
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    # Create and fit model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    # evaluate model
    yhat = model.predict(X_test)
    # store values
    y_true.append(y_test[0])
    y_pred.append(yhat[0])

y_pred = np.array(y_pred)
y_true = np.array(y_true)
print('-----------------First Part-----------------')
# calculate accuracy
acc = accuracy_score(y_true, y_pred)
print('Accuracy: %.5f' % acc)

cnf_matrix = confusion_matrix(y_true, y_pred)


TN = cnf_matrix[0][0]
FN = cnf_matrix[1][0]
TP = cnf_matrix[1][1]
FP = cnf_matrix[0][1]

print('True negatives: %.1f' % TN)
print("False positives: %.1f" % FP)
print("False negatives: %.1f" % FN)
print("True positives: %.1f" % TP)



##################### Second Part #####################

print('-----------------Second Part-----------------')

data = pd.read_csv('algo_performance.csv')


stat, p = friedmanchisquare(data.C4, data.NN, data.NaiveBayes, data.Kernel, data.CN2)

print('Statistics= %.3f, p= %f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
