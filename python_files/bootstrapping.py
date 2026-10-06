# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:30:49 2020

@author: DH
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import random as rn
import os


# we check the path and open the data set
print(os.getcwd())
data = pd.read_csv("hard_drive_failure_data.csv",sep=';' ) 
# we have a dataframe
print(type(data))
# and print the first line
print(data.head())
# we retrieve the lifetime & convert them into an array
datn = data['lifetime'].values
# we plot the empirical distribution
plt.hist(datn, 50, density=1,facecolor='g', alpha=0.75,edgecolor='b')
plt.xlabel('days')
plt.ylabel('Probability')
plt.title('Hard drive lifetimes')
# we estimate beta by log-likelihood maximization
bet=sc.expon.fit(data=datn,scale=1)
# and compare with the moment matching estimator
betm=np.mean(datn)
# x is the linear space
x  = np.linspace(min(datn),max(datn),100)
fx = sc.expon.pdf(x,scale=bet[1])
plt.plot(x,fx,'r',lw=2)
plt.grid(True)
plt.legend(['Empirical distribution','Fitted distribution'])

###############################################################################
# Function computing the log-likelihood
###############################################################################
def log_likelihood_expon(data, beta):
    L = []
    for x in data:
        y =  sc.expon.pdf(x,scale=beta)
        L.append(np.log(y))
    return np.sum(L)  

ll=log_likelihood_expon(datn, bet[1])
print(ll)
###############################################################################
# Bootstrapping
###############################################################################
M=10000
bet=[]
for m in np.arange(0,M):
    datm = rn.choices(population=datn, k=len(datn))
    betm = sc.expon.fit(data=datm,scale=1)
    bet.append(betm[1])

plt.hist(bet, 50, density=1,facecolor='g', alpha=0.75,edgecolor='b')
plt.xlabel('beta estimator')
plt.ylabel('Probability')
plt.title('Estimator distribution')
plt.grid(True)

beta_avg = np.mean(bet)
beta_std = np.std(bet)
betal    = np.quantile(bet,q=0.025)
betau    = np.quantile(bet,q=0.975)

print('Beta estimate :' + str(round(beta_avg,2)))
print('Beta standard deviation :' + str(round(beta_std,2)))
print('Beta 2,5%  quantile :' + str(round(betal)))
print('Beta 97,5% quantile :' + str(round(betau)))

###############################################################################
# Bootstrapping : same exercise but we trim the data set
# we work with 1000 observations instead of 2987
###############################################################################
dats = datn[0:1000]
M    = 10000
bet=[]
for m in np.arange(0,M):
    datm = rn.choices(population=dats, k=len(dats))
    betm = sc.expon.fit(data=datm,scale=1)
    bet.append(betm[1])

plt.hist(bet, 50, density=1,facecolor='g', alpha=0.75,edgecolor='b')
plt.xlabel('beta estimator')
plt.ylabel('Probability')
plt.title('Estimator distribution')
plt.grid(True)

beta_avg = np.mean(bet)
beta_std = np.std(bet)
betal    = np.quantile(bet,q=0.025)
betau    = np.quantile(bet,q=0.975)

print('Beta estimate :' + str(round(beta_avg,2)))
print('Beta standard deviation :' + str(round(beta_std,2)))
print('Beta 2,5%  quantile :' + str(round(betal)))
print('Beta 97,5% quantile :' + str(round(betau)))



