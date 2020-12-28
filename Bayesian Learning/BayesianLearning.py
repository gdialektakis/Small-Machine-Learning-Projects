from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

"""
 I have tried many values of alpha in the range [0,3] 
 and Ι figured out that the bigger alpha is the smaller f1 becomes.
 For this reason, I select alpha=3.0 so that f1 < 70%, as asked.
 """
model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=3.0))
model.fit(train.data, train.target)
predicted_labels = model.predict(test.data)

f1 = '{0:.5f}'.format(metrics.f1_score(test.target, predicted_labels, average='macro'))
accuracy = '{0:.5f}'.format(metrics.accuracy_score(test.target, predicted_labels))
recall = '{0:.5f}'.format(metrics.recall_score(test.target, predicted_labels, average='macro'))
precision = '{0:.5f}'.format(metrics.precision_score(test.target, predicted_labels, average='macro'))

# setting the size of the figure the heatmap will be printed
figure(figsize=(11.2, 7))

conf_matrix = confusion_matrix(test.target, predicted_labels)
sb.heatmap(conf_matrix.T, square=False, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names, robust=True)

plt.title("Multinomial NB - Confusion Matrix (a = 3.0) [Prec = " + str(precision) +
          ", Rec = " + str(recall) + ", F1 = " + str(f1) + ", Acc = " + str(accuracy) + "]")

plt.show()
