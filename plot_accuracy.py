import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pylab import *


# F2 accuracy scores from decision tree classifier using different number of topics from LDA model as classifier features
accuracy = [[0.63725490196078438, 500], [0.59139784946236562, 510], [0.17045454545454547, 520], [0.625, 530],
			 [0.53191489361702127, 540], [0.74468085106382975, 550], [0.38888888888888884, 560], [0.73684210526315785, 570], 
			 [0.57894736842105265, 580], [0.47368421052631582, 590], [0.69148936170212771, 600], [0.11904761904761904, 5], [0.0, 30], 
			 [0.0, 55], [0.061728395061728399, 80], [0.061728395061728399, 105], [0.059523809523809521, 205], [0.12048192771084337, 305], 
			 [0.55555555555555558, 405], [0.32967032967032966, 505], [0.625, 620], [0.69306930693069302, 670], 
			 [0.67307692307692324, 720], [0.75757575757575757, 770], [0.75471698113207553, 820], [0.5670103092783505, 870], 
			 [0.7142857142857143, 920], [0.74766355140186924, 970], [0.59405940594059392, 1000], [0.57291666666666674, 1050], 
			 [0.33333333333333331, 1100], [0.75, 1150], [0.57894736842105265, 1200]]

y, x = zip(*accuracy)

# fit polynomial regression line to x, y
a, b, c, d = polyfit(x,y,3)
yp = polyval([a, b, c, d], x)


# plot
plt.plot(x, yp, linewidth=3, c='green', alpha=0.5)
plt.scatter(x,y, s=100, alpha=0.75, c='steelblue')
plt.xlabel('Number of Topics')
plt.ylabel('Accuracy')
plt.show()

