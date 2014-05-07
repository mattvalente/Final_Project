from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel, lsimodel
import pickle
import timeit
import numpy as np
from sklearn import svm


# ***************************************
# prepare corpus for lda training
# ***************************************

def create_list(books):
  '''turns reviews file into a list
    books ex: booktenthousand.txt'''
  reviews = []
  with open(('../%s' % books), 'r') as f:
    for line in f:
      reviews.append(line)
  return reviews

def stop_words(stoptext):
  '''each review becomes a list of words
    stop words are removed from each review list'''
  with open (('../%s' % stoptext), 'r') as f:
      data= f.read().replace('\n', '')
  stop_list = set(data.split())
  texts = [[word for word in review.split() if word not in stop_list]
  for review in reviews]
  return texts    # format: [['how', 'now', 'brown', 'cow'], ['apple', 'banana']]

def remove_rare(texts):
  '''removes all words that only appear once'''
  all_tokens = sum(texts, [])		# creates 1 list of all words in all reviews
  tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
  texts = [[word for word in text if word not in tokens_once]
           for text in texts]
  return texts    # format: ['how', 'now', 'brown', 'cow', 'apple', 'banana']

def lda_objects(texts):
  '''creates dictionary and corpus objects
    which are used as input to lda model'''
  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
  return dictionary, corpus


# ***************************************
# save objects to file for later use
# ***************************************

def save_thing(thing, filename):
  '''saves object in file for
    later use'''
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
  '''takes a list of labeled reviews and returns
    a list of review topic vectors and a separate
    list of labels'''
  X = []    # list of reviews as topic vectors
  for review in reviews:
    vector = dictionary.doc2bow(review[0].lower().split())
    review_lda = lda[vector]  # type: list of tuples: [(topic_#, probability), ]
    print review_lda
    topic_vector = [tup[1] for tup in review_lda]  # vector of topic probabilities
    X.append(topic_vector)
  y = [review[1] for review in reviews]
  return X, y


# ***************************************
# classifier
# ***************************************

# fit classifier
# clf = svm.SVC()
# clf.fit(X, y)

# predict
# clf.predict(X)  # input format: ([ [], [], ...])


# time tests; these run if you run script as main
'''
timeit.timeit("create_list()", setup="from __main__ import test", number=1)
timeit.timeit("stop_words()", setup="from __main__ import test", number=1)
timeit.timeit("remove_rare()", setup="from __main__ import test", number=1)
timeit.timeit("lda_objects()", setup="from __main__ import test", number=1)
'''

if __name__ == '__main__':
    main()
