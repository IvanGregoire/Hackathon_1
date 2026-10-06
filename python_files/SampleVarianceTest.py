# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:01:21 2020

@author: DH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import statsmodels.api as sm
import os

# we check the path and open the data set
print(os.getcwd())
data = pd.read_csv("bottles.csv",sep=';' ) 
# we have a dataframe, we convert it in an array and just
datn = data.values
# X volumes in ml of soda
X    = datn[:,0]
n    = len(X)
# we start with descriptive statistics
Stat=sc.describe(X)
# we print the mean & variance
print([Stat.mean , Stat.variance])


#Let's check ithat the assumption variance=20^2
#cannot be rejected, first we calculate the statistics
sg0=20 
Tx= (n-1)*Stat.variance/sg0**2
# we compare it to percentiles of a chi2 distribution
alpha = 0.05
t_u   = sc.chi2.ppf(q=1-alpha/2,df=n-1)
#we see that Tx is in the 2.5% and 97.5% interval
#of a chi-square
print([Tx,t_u])
#The p-value is
pval = 1-sc.chi2.cdf(Tx,df=n-1)
print(pval)

#plot of the p-value
x  = np.linspace(sc.chi2.ppf(0.001,df=n-1),
                 sc.chi2.ppf(0.999,df=n-1), 100)
fx = sc.chi2.pdf(x,df=n-1)

xa  = np.linspace(sc.chi2.ppf((1-alpha),df=n-1),
                 sc.chi2.ppf(0.999,df=n-1), 100)
fxa = sc.chi2.pdf(xa,df=n-1)
fxa[0]=0
fxa[99]=0

xp  = np.linspace(sc.chi2.ppf((1-pval),df=n-1),
                 sc.chi2.ppf(0.999,df=n-1), 100)
fxp = sc.chi2.pdf(xp,df=n-1)
fxp[0]=0
fxp[99]=0


plt.plot(x, fx,'b-', lw=1, alpha=0.6)
plt.title('Example 1, sample variance test')
plt.grid(True)
plt.fill(xp, fxp, "g")
plt.fill(xa, fxa, "r")
plt.legend(['pdf','p-value','Reject H0 at 5%'])
