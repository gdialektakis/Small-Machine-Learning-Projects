# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn import metrics

random.seed = 42
np.random.seed(666)



# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
titanic = pd.read_csv("titanic.csv")
y = titanic.Survived

titanic.drop(['PassengerId', 'Sex', 'Age', 'Cabin', 'Survived', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(titanic, y, test_size=0.2, random_state=0)

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================

scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
imputer = KNNImputer(n_neighbors=3)
imputer.fit_transform(titanic)

recall_score_array = []
precision_score_array = []
accuracy_score_array = []
f1_score_array = []

# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================
weights = 'distance'
for i in range(199):
    knn_classifier = KNeighborsClassifier(n_neighbors=i+1, weights=weights, p=1)
    knn_classifier.fit(X_train, y_train)
    y_predicted = knn_classifier.predict(X_test)

    # Storing metrics
    recall_score_array.append(metrics.recall_score(y_test, y_predicted))
    precision_score_array.append(metrics.precision_score(y_test, y_predicted))
    accuracy_score_array.append(metrics.accuracy_score(y_test, y_predicted))
    f1_score_array.append(metrics.f1_score(y_test, y_predicted))

f1_max = max(f1_score_array)
f1max_index = f1_score_array.index(max(f1_score_array))
accuracy = accuracy_score_array[f1max_index]
precision = precision_score_array[f1max_index]
recall = recall_score_array[f1max_index]

f1_no_impute = f1_score_array
numOfNeighbours = f1max_index + 1
print("F1: %2f" % f1_max)
print("Accuracy: %2f" % accuracy)
print("Precision: %2f" % precision)
print("Recall: %2f" % recall)
print("Neighbours: %f" % numOfNeighbours)

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
plt.title('k-Nearest Neighbors (Weights=<uniform>, Metric = <Minkowski>, p = <1>)')
#plt.plot(f1_impute, label='with impute')
plt.plot(f1_no_impute, label='without impute')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.show()

