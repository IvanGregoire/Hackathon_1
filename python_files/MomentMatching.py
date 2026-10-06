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
# we have a dataframe, we convert it in an array and just
# store the durations of assembly for a team of 3 workers in X
datn = data.values
X    = datn[:,0]
#Histogram of assembling times, this is the empirical pdf
n, bins, patches = plt.hist(X, 30, density=1,facecolor='g', alpha=0.75)
plt.xlabel('Minutes')
plt.ylabel('Probability')
plt.title('Assembling time 3 workers')
plt.grid(True)

#we calculate the empirical mean and variance
muX=np.mean(X)
sgX=np.std(X)
#     can we fit an exponential random variable?
# using the method of moments, we find beta = 1/muX
bet = muX
sc.expon.ppf(0.01,scale=bet)
x = np.linspace(sc.expon.ppf(0.01,scale=bet),
                 sc.expon.ppf(0.9,scale=bet), 100)
plt.plot(x, sc.expon.pdf(x,scale=bet),
        'r-', lw=2, alpha=0.6, label='expon pdf')

#     can we fit a Gamma random variable?
alp = muX**2/sgX**2
bet = sgX**2/muX
plt.plot(x, sc.gamma.pdf(x,a=alp,scale=bet),
        'b-', lw=2, alpha=0.6, label='Gamma pdf')

#     can we fit a Normal random variable?
plt.plot(x, sc.norm.pdf(x,loc=muX,scale=sgX),
        'k-', lw=2, alpha=0.6, label='Norm pdf')

plt.legend(['Histogram', 'Expo','Gamma','Normal'])
