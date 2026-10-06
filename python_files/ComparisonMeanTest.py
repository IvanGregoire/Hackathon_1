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
Xb1   = np.mean(X1)
Xb2   = np.mean(X2)
S1    = np.std(X1,ddof=1) 
S2    = np.std(X2,ddof=1)
Spool = np.sqrt(((n-1)*S1**2+(n-1)*S2**2)/(n+n-2))
Tx    = (Xb1-Xb2)/(Spool*np.sqrt(1/n+1/n))
# we compare it to percentiles of a t distribution
alpha = 0.05
t_l   = sc.t.ppf(q=alpha/2,df=n+n-2)
t_u   = sc.t.ppf(q=1-alpha/2,df=n+n-2)
#we see that Tx is in the 2.5% and 97.5% interval of the Student's t
print([Tx,t_l,t_u])
pval = 2*sc.t.cdf(-np.abs(Tx),df=n+n-2)
print(pval)

###############################################################################
# The same computation can be done with the library scipy
# command: ttest_ind
###############################################################################

Txbis , pvalbis=sc.ttest_ind(X1, X2, axis=0, equal_var=True)
print(Txbis,pvalbis)

###############################################################################
# Same computation with the library statsmodels
###############################################################################
import statsmodels.stats.weightstats as sm
# X1 and X2 volumes in ml of soda, machine 1 and 2
X1    = sm.DescrStatsW(datn[:,0])
X2    = sm.DescrStatsW(datn[:,1])
n     = datn.shape[0]

Ttest= sm.CompareMeans(X1,X2)
print(Ttest.tconfint_diff(usevar='pooled'))
print(Ttest.summary(usevar='pooled'))
print(Ttest.ttest_ind(usevar='pooled',value=0))

