import numpy.random as r


#sample = r.random_integers(0, 12886487, 10)
sample = list(r.choice(100, 10, replace=False))
sample.sort()
#print sample
l = []
with open('../a.txt', 'r') as f:
  i = 0
  for line in f:
    if i in sample:
      l.append([line, i])
      sample.pop(0)
    i += 1

#print len(l)
#print l
with open('../bookthousand.txt', 'w') as f:
  [f.write(i) for i in l]
