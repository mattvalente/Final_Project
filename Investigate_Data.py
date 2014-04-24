from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel

# Below opens the file. As it stands, I'm just opening a selected number of review_lines.
# I want to be able to select a random sample, but I think we want to have the same random sample.
# I propose we do the following:
	# 1. Open the file
	# 2. Read every 11th line starting at row 9 (these are the reviews)
	# 3. Save all the lines that we read as a separate file
	# 4. Create a random sample from that file (we need to decide on size) and save that
	# 5. We'll work on the file we created in step 4 so that we're always using the same random sample

import itertools
documents = []
with open('../Arts.txt', 'r') as f:
    review_lines = itertools.islice(f, 9, 20000, 11)
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
lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10)
corpus_lda = lda[corpus]

#for l,t in itertools.izip(corpus_lda,corpus):
  #print l,"#",t

for i in range(0, lda.num_topics-1):
	print lda.print_topic(i)

