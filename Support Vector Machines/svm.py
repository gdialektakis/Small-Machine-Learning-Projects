import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics, model_selection, preprocessing

data = pd.read_csv("creditcard.csv")

y = data["Class"]
X = data.drop("Class", axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

tests = [
    [0.1, "poly", 0.2, 2],
    [10, "poly", 6, 5],
    [0.1, "rbf", 0.3, 0],
    [10, "rbf", 5, 0],
    [0.1, "sigmoid", 0.5, 0],
    [10, "sigmoid", 2, 0],
    [100, "sigmoid", 5, 0]
]
i = 0
for test in tests:
    i += 1
    svc = SVC(kernel=test[1], gamma=test[2], C=test[0], degree=test[3], max_iter=10000)
    svc.fit(x_train, y_train)
    y_predicted = svc.predict(x_test)
    print("Test %d" % i)
    print("------------- SVM Evaluation with parameters: -------------\n")
    parameters_str = "C = " + str(test[0]) + ", Kernel = " + str(test[1]) + ", Gamma = " + str(test[2])
    if test[1] == 'poly':
        parameters_str += ",  Degree = " + str(test[3])

    print(parameters_str)

    accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average="macro")
    precision = metrics.precision_score(y_test, y_predicted, average="macro")
    f1 = metrics.f1_score(y_test, y_predicted, average="macro")


    print("Accuracy: %2f" % accuracy)
    print("Precision: %2f" % precision)
    print("Recall: %2f" % recall)
    print("F1: %2f" % f1)
    print("\n")

