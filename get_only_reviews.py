import itertools
from sys import argv
'''
writes all reviews to specified file
must input filenames in command line
	$ python reviews.py input_file.txt outputfile.txt
'''

input_file, output_file = argv[1], argv[2]
intext = '../' + input_file
outtext = '../' + output_file

reviews = []									# list of reviews
with open(intext, 'r') as f:
    review_lines = itertools.islice(f, 9, None, 11)
    for line in review_lines:
    	reviews.append(line[13:].lower())				# line[13:] cuts out 'review/text: ' from each line


with open(outtext, 'w') as f:
	[f.write(review) for review in reviews]
