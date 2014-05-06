from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel, lsimodel
import pickle
import time



#start = time.clock()
reviews = []
with open('../booktenthousand.txt', 'r') as f:
	for line in f:
		reviews.append(line)
#end = time.clock()
#print 'reviews!'
#print 'time: ', (end -start)


#start = time.clock()
# remove stop words
with open ('Stop_Words.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')
stop_list = set(data.split())
texts = [[word for word in review.split() if word not in stop_list]			# splits words in each review and...
for review in reviews]														# takes stop words out of each review
																			# [['how now brown cow'], ['apple banana']] --> [['how', 'now', 'brown', 'cow'], ['apple', 'banana']]
#end = time.clock()
#print 'stopwords!'
#print 'time: ', (end - start)


#start = time.clock()
# remove words that appear only once
all_tokens = sum(texts, [])		# creates 1 list of all words in all reviews
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]
#end = time.clock()
#print 'remove words!'
#print 'time: ', (end - start)


#start = time.clock()
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#end = time.clock()
#print 'corpus!'
#print 'time :', (end - start)


#start = time.clock()
# saves dictionary and corpus objects to file
with open('../dfile.obj', 'w') as f:
	pickle.dump(dictionary, f)
with open('../cfile.obj', 'w') as g:
	pickle.dump(corpus, g)
#end = time.clock()
#print 'pickle yeah!'
#print 'time: ', (end - start)



# saves dictionary and corpus objects for later use
with open('../dfile.obj', 'r') as f:
	dictionary = pickle.load(f)
with open('../cfile.obj', 'r') as g:
	corpus = pickle.load(g)

#print 'pickled!'
# LDA
#start = time.clock()
lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10)
#end = time.clock()
#print 'lda time: ', (end - start)


# saves lda model
lda.save('ldamodel')


#lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=3)
#corpus_lda = lda[corpus]


'''
for i in range(0, lda.num_topics):
	print lda.print_topic(i)
'''
