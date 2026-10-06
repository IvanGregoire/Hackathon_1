# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:05:32 2020

@author: DH
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
##############################################################################
# We compare a binomial pmf to its Gaussian approximation
# Here the Gaussian approximation is reliable
##############################################################################
n = 10
p = 0.5
k = range(0,n)
Pk= sc.binom.pmf(k, n, p)
# Gaussian approx
mu = n*p
sg = np.sqrt(n*p*(1-p))
x  = np.linspace(0,n,100)
fx = sc.norm.pdf(x,loc=mu,scale=sg)

#bar plot & plot of pdf
plt.subplot(121)
plt.bar(k,Pk,facecolor='g')
plt.title('Binomial pmf')
plt.grid(True)

plt.plot(x,fx,'r')
plt.grid(True)
plt.legend(['n=10, p=0.5'])

##############################################################################
# We compare a binomial pmf to its Gaussian approximation
# Here this is not good since the Gaussian allows for negative
# values with a significant probability
##############################################################################
n = 10
p = 0.1
k = range(0,n)
Pk= sc.binom.pmf(k, n, p)

plt.subplot(122)

mu = n*p
sg = np.sqrt(n*p*(1-p))
x  = np.linspace(-n/2,n,100)
fx = sc.norm.pdf(x,loc=mu,scale=sg)

plt.plot(x,fx,'r')
plt.grid(True)
plt.legend(['n=10, p=0.1'])

plt.bar(k,Pk,facecolor='g', )
plt.title('Binomial pmf')
plt.grid(True)


#