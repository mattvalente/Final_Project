import numpy.random as r

# create sample list
sample = list(r.choice(12886487, 10000, replace=False))
sample.sort()


l = []        # list of sampled reviews


# put sampled reviews into list l
with open('../bookreviews.txt', 'r') as f:
  i = 0
  for line in f:
    if i in sample:
      l.append([line])
      sample.pop(0)
    if i == 1000000:
        print '1 million!'      # to check progress
    if i == 5000000:
      print '5 million yeah!'
    i += 1


# write list l to file
with open('../booktenthousand.txt', 'w') as f:
  [f.write(i[0]) for i in l]
