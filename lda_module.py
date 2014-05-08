from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel, lsimodel
import pickle
import timeit
import numpy as np
from sklearn import svm, metrics
import re


# ***************************************
# prepare corpus for lda training
# ***************************************


def remove_stuff(stoptext, reviews_txt):
  '''removes stopwords and rare words, and
    prepares text for lda_objects function'''
  # read in reviews text as a string
  with open(('../%s.txt' % reviews_txt), 'r') as f:
    string = f.read()
  # read in stopwords text as a list of words
  with open (('../%s.txt' % stoptext), 'r') as f:
      data= f.read().replace('\n', '')
  stop_list = set(data.split())
  # remove punctuation
  only_words = re.findall(r'\w+', string,flags = re.UNICODE | re.LOCALE)  # returns set of words
  # remove stop words
  remove_stops = [word for word in only_words if word not in stop_list]
  tokens_once = set(word for word in set(remove_stops) if remove_stops.count(word) == 1)
  # remove words that only appear once
  texts = [[word] for word in remove_stops if word not in tokens_once]
  return texts    # format: ['how', 'now', 'brown', 'cow', 'apple', 'banana']


def lda_objects(texts):
  '''creates dictionary and corpus objects
    which are used as input to lda model'''
  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
  return dictionary, corpus

# ***************************************
# prepare train and test set for classification
# ***************************************

def stops(stoptext, reviews_txt):
  '''removes stop words in train/test sets
    and returns as a list of reviews'''
  with open(('../%s.txt' % reviews_txt), 'r') as f:
    reviews = [[word for word in re.findall(r'\w+', line,flags = re.UNICODE | re.LOCALE)] for line in f]  # [['how', 'now'], ['brown', 'cow']]
  with open (('../%s.txt' % stoptext), 'r') as f:
    data= f.read().replace('\n', '')
  stop_list = set(data.split())
  reviews = [[word for word in review if word not in stop_list] for review in reviews]
  reviews = [' '.join(review) for review in reviews]
  return reviews    # format ['review1', 'review2', ...]


# test.txt line_number == 80
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

# lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10, gamma_threshold=None)

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
# clf = svm.SVC()
# clf.fit(X, y)

# predict
# clf.predict(X)  # input format: ([ [], [], ...])

# ***************************************
# optimize number of topics for lda
# ***************************************

def optimize_lda(corpus, dictionary, train, test, topics=10, max_topics=200): 
  '''runs lda with increasing number of topics to optimize
    number of topics; runs svm classifier with each new lda lda model
    and adds error to error_list'''
  error_list = []
  while topics <= max_topics:
    lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=topics)
    print 'lda!'
    X, y_true = topic_vector(train, lda, dictionary)
    classifier = svm.SVC()
    classifier.fit(X, y_true)
    print 'classified!'
    testX, testy = topic_vector(test, lda, dictionary)
    y_pred = classifier.predict(testX)
    print 'predicted!'
    error = metrics.fbeta_score(testy, y_pred, beta=2)    # F2 Score
    error_list.append([error, topics])
    print 'topics! ', topics
    topics += 5
  return error_list

# time tests; these run if you run script as main
'''
timeit.timeit("create_list()", setup="from __main__ import test", number=1)
timeit.timeit("stop_words()", setup="from __main__ import test", number=1)
timeit.timeit("remove_rare()", setup="from __main__ import test", number=1)
timeit.timeit("lda_objects()", setup="from __main__ import test", number=1)
'''

if __name__ == '__main__':
    main()
