import lda_module as m
from sklearn.cross_validation import cross_val_score
from sklearn import tree, svm, metrics
import time
import matplotlib.pyplot as plt  # 2D plotting
import pylab as pl
from sklearn.externals.six import StringIO  


# load data
dictionary = m.load_thing('dictionary')
corpus = m.load_thing('corpus')
train = m.load_thing('train_labeled')
test = m.load_thing('test_labeled')
#lda = m.load_thing('lda_tenthousand')
lda = m.load_thing('lda200')

#print m.optimize_lda(corpus, dictionary, train, test, max_topics=15)
X, y = m.topic_vector(train, lda, dictionary)
testX, testy  = m.topic_vector(train, lda, dictionary)
#print testy



classifier = tree.DecisionTreeClassifier(criterion='gini',max_depth=7,random_state = 9999)
my_tree = classifier.fit(X, y)
probas_ = classifier.fit(X, y).predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testy, probas_[:, 1])
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve DT: %f" % roc_auc)

# Compute confusion matrix
y_pred_DT = classifier.fit(X, y).predict(testX)
cm_DT = metrics.confusion_matrix(testy, y_pred_DT)
print("Confusion Matrix DT:",cm_DT)	

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve Decision Tree (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic Decision Tree')
pl.legend(loc="lower right")
pl.show()	
pl.savefig('../ROC_DecisionTree.pdf')
	
with open('../DecisionTree.dot', 'w') as f:
    f = tree.export_graphviz(my_tree, out_file=f)#

# use dot -Tpdf ../DecisionTree.dot -o ../DecisionTree.pdf to export .dot file as pdf
