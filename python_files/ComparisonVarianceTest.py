# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:01:21 2020

@author: DH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import os

# we check the path and open the data set
print(os.getcwd())
data = pd.read_csv("bottles2D.csv",sep=';' ) 
# we have a dataframe, we convert it in an array and just
datn = data.values
n  = datn.shape[0]
# X1 and X2 volumes in ml of soda, machine 1 and 2
X1 = datn[:,0]
X2 = datn[:,1]
###############################################################################
# we check the equality of means mu1-mu2=0, so delta=0
# we compute the T statistics and the p-value
###############################################################################
S1    = np.std(X1,ddof=1) 
S2    = np.std(X2,ddof=1)
Tx     =S1**2/S2**2
# we compare it to percentiles of a t distribution
alpha = 0.05
f_u   = sc.f.ppf(q=1-alpha,dfn=n-1, dfd=n-1)
#we see that Tx is in the 2.5% and 97.5% interval of the Student's t
print([Tx,f_u])
pval = 1-sc.f.cdf(Tx,dfn=n-1 , dfd=n-1)
print(pval)

###############################################################################
# The command Bartlett(.,.) performs a comparison of variances
# Be careful, it test the two-sided assumption: i.e.
# HO : sigma1 = sigma2  H1 :  sigma1 <>sigma2
###############################################################################

Txbis , pvalbis=sc.bartlett(X1, X2)
print(Txbis,pvalbis)
# we see that pval*2 = pvalbis because this is a two-sided test instead
# of a one sided test.

#plot of the p-value
x  = np.linspace(sc.f.ppf(0.001,dfn=n-1 , dfd=n-1),
                 sc.f.ppf(0.999,dfn=n-1 , dfd=n-1), 100)
fx = sc.f.pdf(x,dfn=n-1 , dfd=n-1)

xa  = np.linspace(sc.f.ppf((1-alpha),dfn=n-1 , dfd=n-1),
                 sc.f.ppf(0.999,dfn=n-1 , dfd=n-1), 100)
fxa = sc.f.pdf(xa,dfn=n-1 , dfd=n-1)
fxa[0]=0
fxa[99]=0

xp  = np.linspace(sc.f.ppf((1-pval),dfn=n-1 , dfd=n-1),
                 sc.f.ppf(0.999,dfn=n-1 , dfd=n-1), 100)
fxp = sc.f.pdf(xp,dfn=n-1 , dfd=n-1)
fxp[0]=0
fxp[99]=0


plt.plot(x, fx,'b-', lw=1, alpha=0.6)
plt.title('Example 1, 2 samples variance test')
plt.grid(True)
plt.fill(xa, fxa, "r")
plt.fill(xp, fxp, "g")

plt.legend(['pdf','Reject H0 at 5%','p-value'])



