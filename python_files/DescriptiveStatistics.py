# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:16:57 2020

@author: DH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import os

# for a list of scipy functions see 
#https://docs.scipy.org/doc/scipy/reference/stats.html

# we check the path and open the data set
print(os.getcwd())
data = pd.read_csv("durationData.csv",sep=';' ) 
# we have a dataframe
print(type(data))
# and print the first line
print(data.head())
print(data['workers3'].head())
# pandas provides basic statistics about data
data = pd.read_csv("durationData.csv",sep=';' ) 
stat = data.describe()
print(stat)
# interquartile range: library scipy.stats
sc.iqr(data[['workers3']])
sc.iqr(data[['workers5']])

# boxplot, we need to convert the data frame into an array
# to display the graph in a window: outils->preferences->
# console Ipython->graphique Sortie : Automatique
datn = data.values
plt.boxplot(datn,labels=['3 workers','5 workers'])
plt.title('Box plot assembling time')
plt.savefig("boxplot.png")

#Histogram of assembling times
plt.subplot(121)
n, bins, patches = plt.hist(datn[:,0], 30, density=1,facecolor='g', alpha=0.75)
plt.xlabel('Minutes')
plt.ylabel('Probability')
plt.title('Assembling time 3 workers')
plt.grid(True)
plt.subplot(122)
n, bins, patches = plt.hist(datn[:,1], 30, density=1,facecolor='g', alpha=0.75)
plt.xlabel('Minutes')
plt.ylabel('Probability')
plt.title('Assembling time 5 workers')
plt.grid(True)

# Computation of statistics
# for a list of numpy functions see 
#https://docs.scipy.org/doc/scipy/reference/stats.html

np.mean(datn[:,0])
np.median(datn[:,0])
np.std(datn[:,0],ddof=1)

# Comparison on two sample of observations, same mean but different variances
mu1, sigma1 = 20, 5
x1          = mu1 + sigma1 * np.random.randn(10000)
mu2, sigma2 = 20, 15
x2          = mu2 + sigma2 * np.random.randn(10000)
# the histogram of the data
n, bins, patches = plt.hist([x1,x2], 30, density=1, 
            color=['r','g'])
plt.ylabel('Probability')
plt.grid(True)
plt.legend(['s=5', 's=15'])

