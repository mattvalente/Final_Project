import lda_module as m 

lda = m.load_thing('lda200')

with open('../200Topics_200words.txt', 'w') as f:
	for i in lda.print_topics(topics=200,topn=5):
		f.write('%s \n' % i) 
