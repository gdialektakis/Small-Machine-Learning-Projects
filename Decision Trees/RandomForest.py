# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, ensemble, model_selection
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================

# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
breastCancer = datasets.load_breast_cancer()

# =============================================================================

# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Also, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)


# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================


# ADD COMMAND TO CREATE RANDOM FOREST CLASSIFIER MODEL HERE
model = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=500)

# =============================================================================

# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN YOUR MODEL HERE

# =============================================================================
model.fit(x_train, y_train)

# Ok, now let's predict the output for the test set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = model.predict(x_test)

# =============================================================================


# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used (for now):
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).

# =============================================================================


# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Prediction on test data")
print("Recall score: %2f" % metrics.recall_score(y_test, y_predicted))
print("Precision score: %2f" % metrics.precision_score(y_test, y_predicted))
print("Accuracy score: %2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 score: %2f" % metrics.f1_score(y_test, y_predicted))
# =============================================================================


# A Random Forest has been trained now, but let's train more models, 
# with different number of estimators each, and plot performance in terms of
# the different metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# After finishing the above plots, try doing the same thing on the train data
# Hint: you can plot on the same figure in order to add a second line.
# Change the line color to distinguish performance metrics on train/test data
# In the end, you should have 4 figures (one for each metric)
# And each figure should have 2 lines (one for train data and one for test data)


# CREATE MODELS AND PLOTS HERE
# arrays to store performance scores, which will be used for plotting
recall_score_array = []
precision_score_array = []
accuracy_score_array = []
f1_score_array = []

recall_score_train_array = []
precision_score_train_array = []
accuracy_score_train_array = []
f1_score_train_array = []


# create 100 trees
for i in range(1, 100):
 model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=i)
 model.fit(x_train, y_train)

 y_predicted = model.predict(x_test)
 y_predicted_train = model.predict(x_train)

 # evaluating on test data
 recall_score_array.append(metrics.recall_score(y_test, y_predicted))
 precision_score_array.append(metrics.precision_score(y_test, y_predicted))
 accuracy_score_array.append(metrics.accuracy_score(y_test, y_predicted))
 f1_score_array.append(metrics.f1_score(y_test, y_predicted))

 # evaluating on training data to measure overfitting
 recall_score_train_array.append(metrics.recall_score(y_train, y_predicted_train))
 precision_score_train_array.append(metrics.precision_score(y_train, y_predicted_train))
 accuracy_score_train_array.append(metrics.accuracy_score(y_train, y_predicted_train))
 f1_score_train_array.append(metrics.f1_score(y_train, y_predicted_train))

# ============================================================================
# plotting

plt.figure()
plt.xlabel('Number of Estimators')
plt.ylabel('Recall')
plt.plot(range(1, 100), recall_score_array, 'b--', label='test')
plt.plot(range(1, 100), recall_score_train_array, 'r:', label='train')
plt.legend(loc='best')

plt.figure()
plt.xlabel('Number of Estimators')
plt.ylabel('Precision')
plt.plot(range(1, 100), precision_score_array, 'b--', label='test')
plt.plot(range(1, 100), precision_score_train_array, 'r:', label='train')
plt.legend(loc='best')
plt.figure()

plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.plot(range(1, 100), accuracy_score_array, 'b--', label='test')
plt.plot(range(1, 100), accuracy_score_train_array, 'r:', label='train')
plt.legend(loc='best')

plt.figure()
plt.xlabel('Number of Estimators')
plt.ylabel('F1')
plt.plot(range(1, 100), f1_score_array, 'b--', label='test')
plt.plot(range(1, 100), f1_score_train_array, 'r:', label='train')

plt.legend(loc='best')
plt.show()


# ============================================================================
