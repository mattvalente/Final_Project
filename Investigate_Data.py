from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel

import itertools
documents = []
with open('../Arts.txt', 'r') as f:
    review_lines = itertools.islice(f, 9, 1000, 11)
    for line in review_lines:
    	documents.append(line[13:len(line)-1].lower())


# remove stop words
with open ('Stop_Words.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')	
stop_list = set(data.split())
#stop_list = set('for a of the and to in i they are these as all with my have on you that on was is we this were it so are not be but will do can'.split())
texts = [[word for word in document.split() if word not in stop_list]
for document in documents]

#print stop_list

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA
lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=5)
corpus_lda = lda[corpus]

#for l,t in itertools.izip(corpus_lda,corpus):
  #print l,"#",t

for i in range(0, lda.num_topics-1):
	print lda.print_topic(i)

