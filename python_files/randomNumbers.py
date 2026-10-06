# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:52:11 2020

@author: DH
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

###############################################################################
# Homemade function to generate random numbers in [0,1]
###############################################################################
def LinearCongruentGenerator(n=1,a=1103515245,c=12345,m=2**31,x0=5):
   prs = [];
   xn = x0;
   for i in range (n):
       xn = (a*xn + c) % m;
       prs.append (xn);
   return prs;

# we run the linear congruent generator
X=LinearCongruentGenerator(n=10000)
# we plot the result and see that X is uniformally distributed on [0,1]
plt.hist(X, 30, density=1,facecolor='cyan',edgecolor = 'red',   alpha=0.6)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('10 000 pseudo rnd numbers')
plt.grid(True)

###############################################################################
#There exists random numbers generators in Python. See library "random"
###############################################################################
import random as rn
#uniform generate a number in (0,1)
print(rn.uniform(0,1))

###############################################################################
# The library numpy has also function to generate a vector of rnd 
# numbers in [0,1]
###############################################################################
X=np.random.random_sample(10000)
plt.close()
plt.hist(X, 30, density=1,facecolor='cyan',edgecolor = 'red',   alpha=0.6)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('10 000 pseudo rnd numbers with numpy')
plt.grid(True)

###############################################################################
# The library scipy has also function to generate a vector of rnd 
# numbers in [0,1]
###############################################################################
X=sc.uniform.rvs(loc=0,scale=1,size=10000)
plt.close()
plt.hist(X, 30, density=1,facecolor='cyan',edgecolor = 'red',   alpha=0.6)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('10 000 pseudo rnd numbers with numpy')
plt.grid(True)


###############################################################################
# The library scipy has also functions to generate random variables:
# Gaussian, chi-square etc....
###############################################################################
mu=25
sg=5
X=sc.norm.rvs(loc=mu,scale=sg,size=1000)
plt.close()
plt.subplot(2,1,1)
plt.hist(X, 30, density=1,facecolor='cyan',edgecolor = 'red',   alpha=0.6)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('1000 normal rnd numbers with numpy')
plt.grid(True)

bet=25
X=sc.expon.rvs(scale=bet,size=1000)

plt.subplot(2,1,2)
plt.hist(X, 30, density=1,facecolor='cyan',edgecolor = 'red',   alpha=0.6)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('1000 Expo rnd numbers with numpy')
plt.grid(True)