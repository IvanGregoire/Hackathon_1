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

# does the machine fill on average 500ml of soda? t-test
# H0 : mu = 500
# H1 : mu not equal to 500
Ttest=sc.ttest_1samp(X, popmean=500)
print(Ttest)

#The p-value is larger than e.g. 5%, we do not reject H0

#Let's check it, first we calculate the statistics 
Tx= (Stat.mean-500)/np.sqrt(Stat.variance/n)
# we compare it to percentiles of a t distribution
alpha = 0.05
t_l   = sc.t.ppf(q=alpha/2,df=n-1)
t_u   = sc.t.ppf(q=1-alpha/2,df=n-1)
#we see that Tx is in the 2.5% and 97.5% interval of the Student's t
print([Tx,t_l,t_u])
#The p-value is
pval = 2 * (1-sc.t.cdf(np.abs(Tx),df=n-1))
print(pval)


#we just plot the p-values

x  = np.linspace(sc.t.ppf(0.001,df=n-1),
                 sc.t.ppf(0.999,df=n-1), 100)
fx = sc.t.pdf(x,df=n-1)

xa  = np.linspace(sc.t.ppf((1-alpha/2),df=n-1),
                 sc.t.ppf(0.999,df=n-1), 100)
fxa    = sc.t.pdf(xa,df=n-1)
fxa[0] =0 ; fxa[99]=0

xb  = np.linspace(sc.t.ppf(0.001,df=n-1),
                 sc.t.ppf(alpha/2,df=n-1), 100)
fxb    = sc.t.pdf(xb,df=n-1)
fxb[0] =0 ; fxb[99]=0

xp  = np.linspace(sc.t.ppf((1-pval/2),df=n-1),
                 sc.t.ppf(0.999,df=n-1), 100)
fxp    = sc.t.pdf(xp,df=n-1)
fxp[0] =0
fxp[99]=0

xpb  = np.linspace(sc.t.ppf(0.001,df=n-1),
                 sc.t.ppf(pval/2,df=n-1), 100)
fxpb    = sc.t.pdf(xpb,df=n-1)
fxpb[0] =0
fxpb[99]=0


plt.plot(x, fx,'b-', lw=1, alpha=0.6, label='Student t pdf')
plt.title('Example 1, sample mean test')
plt.grid(True)
plt.fill(xp, fxp, "g")
plt.fill(xa, fxa, "r")
plt.fill(xpb, fxpb, "g")
plt.fill(xb, fxb, "r")
plt.legend(['pdf','p-value','Reject H0 at 5%'])

