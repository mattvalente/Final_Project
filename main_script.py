from gensim import corpora, models, similarities
from gensim.models import ldamodel
import pickle
import timeit
import time
import numpy as np
from sklearn import tree, metrics
import re


# ***************************************
# prepare corpus for lda training
# ***************************************



def text_prep(stoptext, reviews_txt):       # use this function instead of stops or remove_stuff
  with open(('../%s.txt' % reviews_txt), 'r') as f:
    documents = [doc for doc in f]      # list of reivews
  # read in stopwords text as a list of words
  with open (('../%s.txt' % stoptext), 'r') as f:
    data= f.read().replace('\n', '')
  stoplist = set(data.split())
  texts = [[word for word in re.findall(r'\w+', document.lower(),flags = re.UNICODE | re.LOCALE) if word not in stoplist] 
        for document in documents]
  # remove words that appear only once
  all_tokens = sum(texts, [])
  tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
  texts = [[word for word in text if word not in tokens_once] 
           for text in texts]
  return texts    # format [['how', 'now', 'brown', 'cow'], ['apples', 'bananas']]

def lda_objects(texts):
  '''creates dictionary and corpus objects
    which are used as input to lda model'''
  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
  return dictionary, corpus

# ***************************************
# prepare train and test set for classification
# ***************************************


# test.txt line_number == 81
# train.txt line_number == 1,000
def labels(reviews, line_number):
  '''takes a list of reviews, and the
    line_number where seller reviews begin
    and returns a list of review-label pairs'''
  reviews = [[review] for review in reviews]  # [['apple banana'], ['porcupine sandwich'], ...]
  i = 1
  for review in reviews:
    if i < line_number:
      review.append(0)
    else:
      review.append(1)
    i += 1
  return reviews




# ***************************************
# save objects to file for later use
# ***************************************

def save_thing(thing, filename):
  '''saves object in file for
    later use; filename must be in quotes'''
  with open(('../%s.obj' % filename), 'w') as f:
    pickle.dump(thing, f)

def load_thing(filename):
  '''loads object as variable: thing
      from file: filename'''
  with open(('../%s.obj' % filename), 'r') as f:
    thing = pickle.load(f)
  return thing

# ***************************************
# train lda model
# ***************************************

# lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10)

# ***************************************
# create review vectors for classification
# from a labeled training set
# ***************************************


def topic_vector(reviews, lda, dictionary):
  '''takes a list of labeled reviews [[review1, label], [review2, label]]
    and returns  a list of review topic vectors
    and a separate list of labels'''
  X = []    # list of reviews as topic vectors
  y = []    # list of labels (0=noseller, 1=seller reviews)
  for review in reviews:
    vector = dictionary.doc2bow(review[0].lower().split())
    review_lda =  lda.__getitem__(vector, eps=0)
    y.append(review[1])
    topic_vector = [tup[1] for tup in review_lda]  # vector of topic probabilities
    X.append(topic_vector)
  return X, y


# ***************************************
# classifier
# ***************************************

# fit classifier
# clf = tree.DecisionTreeClassifier()
# clf.fit(X, y)

# predict
# clf.predict(X)  # input format: ([ [], [], ...])

# ***************************************
# optimize number of topics for lda
# ***************************************

def optimize_lda(corpus, dictionary, train, test, topics=5, max_topics=1200):  # train and test set must be lists of review-label pairs
  '''runs lda with increasing number of topics to optimize
    number of topics; runs svm classifier with each new lda lda model
    and adds error to error_list'''
  accuracy_list = []
  while topics <= max_topics:
    start_time = time.clock()
    lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=topics)
    print 'lda !'
    x_train, y_train = topic_vector(train, lda, dictionary)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    print 'classified!'
    x_test, y_test = topic_vector(test, lda, dictionary)
    y_pred = list(classifier.predict(x_test)) 
    print 'predicted!'
    accuracy = metrics.fbeta_score(y_test, y_pred, beta=2)    # F2 Score
    accuracy_list.append([accuracy, topics])
    print 'accuracy! ', accuracy 
    #confusion = metrics.confusion_matrix(y_test, y_pred)
    #print 'accuracy ', metrics.accuracy_score(y_test, y_pred)
    #print 'precision ', metrics.precision_score(y_test, y_pred)
    #print 'recall ', metrics.recall_score(y_test, y_pred)
    #topics += 50
    if topics < 100:
      topics += 25
    else:
      topics += 100
    end_time = time.clock()
    print ('lda_%s time: %s' % (topics, (end_time - start_time)))
    #print y_test
    #print y_pred
  #save_thing(accuracy_list, 'accuracy')
  return accuracy_list




if __name__ == '__main__':
    main()

# time tests; these run if you run script as main

#timeit.timeit("text_prep()", setup="from __main__ import test", number=1)
#timeit.timeit("topic_vector()", setup="from __main__ import test", number=1)
#timeit.timeit("optimize_lda()", setup="from __main__ import test", number=1)
