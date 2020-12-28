from sklearn import datasets, metrics, ensemble, model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

X = pd.read_csv("HTRU_2.csv")
y = X.Pulsar
X.drop(['Pulsar'], axis=1, inplace=True)


forest = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=500)


####################### Excluding a feature from the dataset and inspecting model performance afterwards #######################

X_F7 = X.drop(['F7'], axis=1)

X_F7_train, X_F7_test, y_train, y_test = train_test_split(X_F7, y, test_size=0.2, random_state=0)

forest.fit(X_F7_train, y_train)

# Ok, now let's predict the output for the test set
# =============================================================================

y_predicted = forest.predict(X_F7_test)
false_positive, true_positive, thresholds = metrics.roc_curve(y_test, y_predicted, drop_intermediate=True)

print("\n")
print ("-------------------Excluding a feature from the dataset and inspecting model performance afterwards-------------------")
print("Prediction on test data without feature 7")
print("Recall score: %2f" % metrics.recall_score(y_test, y_predicted))
print("Precision score: %2f" % metrics.precision_score(y_test, y_predicted))
print("Accuracy score: %2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 score: %2f" % metrics.f1_score(y_test, y_predicted))
print('AUC: %2f' % metrics.auc(false_positive, true_positive))
print('ROC AUC: %2f' % metrics.roc_auc_score(y_test, y_predicted))

fig, ax = plt.subplots()
ax.plot(false_positive, true_positive)
ax.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='baseline',
         linestyle='--')
plt.title('Receiver Operating Characteristic Curve', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.legend(fontsize=12)
plt.show()


################################ Applying PCA ##############################

pca = PCA(n_components=4)
newX = pca.fit_transform(X)

newX_train, newX_test, y_train, y_pca_test = train_test_split(newX, y, test_size=0.2, random_state=0)

forest.fit(newX_train, y_train)

# Ok, now let's predict the output for the test set
# =============================================================================

y_pca_predicted = forest.predict(newX_test)
false_positive, true_positive, thresholds = metrics.roc_curve(y_pca_test, y_pca_predicted, drop_intermediate=True)

print("\n")
print ("-------------------Model performance after PCA-------------------")

print("Recall score: %2f" % metrics.recall_score(y_pca_test, y_pca_predicted))
print("Precision score: %2f" % metrics.precision_score(y_pca_test, y_pca_predicted))
print("Accuracy score: %2f" % metrics.accuracy_score(y_pca_test, y_pca_predicted))
print("F1 score: %2f" % metrics.f1_score(y_pca_test, y_pca_predicted))
print('AUC: %2f' % metrics.auc(false_positive, true_positive))
print('ROC AUC: %2f' % metrics.roc_auc_score(y_test, y_predicted))

fig, ax = plt.subplots()
ax.plot(false_positive, true_positive)
ax.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='baseline',
         linestyle='--')
plt.title('Receiver Operating Characteristic Curve after PCA', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.legend(fontsize=12)
plt.show()


####################### Feature importances with forests of trees #######################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
forest.fit(X_train, y_train)

# Ok, now let's predict the output for the test set
y_predicted = forest.predict(X_test)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("\n")
# Print the feature ranking
print("-------------------Feature ranking-------------------")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


####################### Keeping only the 4 best features based on the previous result #######################
print("--------------Keeping only the 4 best features based on the previous result--------------")
X = X.drop(['F1', 'F7', 'F6', 'F4'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

forest.fit(X_train, y_train)

# Ok, now let's predict the output for the test set
# =============================================================================

y_predicted = forest.predict(X_test)
print("\n")

print("Model evaluation without feature with 4 most important features")
print("Recall score: %2f" % metrics.recall_score(y_test, y_predicted))
print("Precision score: %2f" % metrics.precision_score(y_test, y_predicted))
print("Accuracy score: %2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 score: %2f" % metrics.f1_score(y_test, y_predicted))

print ("-------------------Model performance with PCA-------------------")

print("Recall score: %2f" % metrics.recall_score(y_pca_test, y_pca_predicted))
print("Precision score: %2f" % metrics.precision_score(y_pca_test, y_pca_predicted))
print("Accuracy score: %2f" % metrics.accuracy_score(y_pca_test, y_pca_predicted))
print("F1 score: %2f" % metrics.f1_score(y_pca_test, y_pca_predicted))