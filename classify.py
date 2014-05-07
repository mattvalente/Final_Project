import lda_module as m
from sklearn import svm


# load data
dictionary = m.load_thing('dfile')
corpus = m.load_thing('cfile')
labeled = m.load_thing('booktrain')
#lda = m.load_thing('ldamodel')
lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10, gamma_threshold=None)

# create topic vectors and labels
X, y = m.topic_vector(labeled, lda, dictionary)

'''
# fit classifier
classifier = svm.SVC()
classifier.fit(X, y)

# predict
r = labeled[0][0]
correct = labeled[0][1]
print 'prediction: ', classifier.predict(r)  # input format: ([ [], [], ...])
print 'correct: ', correct
'''
