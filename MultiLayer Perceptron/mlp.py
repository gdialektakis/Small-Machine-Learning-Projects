from sklearn import model_selection, metrics, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer

# load dataset
breastCancer = load_breast_cancer()

X = breastCancer.data
y = breastCancer.target

# split dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
# scale it in range (0, 1)
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# experimental parameters
parameters = [
    [10, "relu", "sgd", 0.0001, "shallow"],
    [20, "tanh", "sgd", 0.0001, "shallow"],
    [20, "tanh", "adam", 0.00001, "shallow"],
    [50, "relu", "adam", 0.00001, "deep"],
    [50, "tanh", "lbfgs", 0.00001, "shallow"],
    [100, "relu", "lbfgs", 0.00001, "deep"]
]

i = 0
for param in parameters:
    i += 1
    if param[4] == "deep":
        mlp = MLPClassifier(random_state=1, max_iter=100, activation=param[1], solver=param[2],
                            tol=param[3], hidden_layer_sizes=(param[0], param[0], param[0]))
    else:
        mlp = MLPClassifier(random_state=1, max_iter=100, activation=param[1], solver=param[2],
                            tol=param[3], hidden_layer_sizes=(param[0]))

    mlp.fit(X_train, y_train)
    mlp.predict_proba(X_test)

    y_predicted = mlp.predict(X_test)

    print("------------------Test %d ------------------" % i)
    print("Evaluation of MLP using parameters:")
    parameters_str = "Activation function = " + str(param[1]) + ", Solver = " + str(param[2]) + ", Tolerance = " + str(
        param[3])

    print(parameters_str)

    print("Recall: %2f" % metrics.recall_score(y_test, y_predicted))
    print("Precision: %2f" % metrics.precision_score(y_test, y_predicted))
    print("Accuracy: %2f" % metrics.accuracy_score(y_test, y_predicted))
    print("\n")
