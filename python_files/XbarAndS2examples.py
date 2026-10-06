# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:34:34 2020

@author: DH
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

###############################################################################
# Example 1, analysis of Xbar and S2
###############################################################################
#we simulate a sample of Gamma with parameters alpha and beta chosen such
#that the mean is equal to 'mu' and the standard deviation is 'sg'
mu= 100
sg= 5
n = 40
X = sc.gamma.rvs(size=n,a= mu**2/sg**2, scale=sg**2/mu) 
#we calculate the mean X^bar and the empirical standard deviation S
Xb = np.mean(X)  
S  = np.std(X)
print([Xb,S])
###############################################################################
# We calculate the probability that   xl < Xb < xu, mu and sg assumed known
###############################################################################
xl=98.4505
xu=101.5495

Proba_xl_xu  = sc.norm.cdf((xu-mu)/(sg/np.sqrt(n)) )       \
                    - sc.norm.cdf((xl-mu)/(sg/np.sqrt(n))) 
print(Proba_xl_xu)

#alternative method
Proba_xl_xu  = sc.norm.cdf(xu , loc=mu, scale=sg/np.sqrt(n) )       \
                    - sc.norm.cdf(xl , loc=mu, scale=sg/np.sqrt(n) ) 
print(Proba_xl_xu)

###############################################################################
# We observe X and known sg. What is the confidence interval of mu
# for a alpha=5% ? (here we assume that mu is unknown)
###############################################################################
alpha = 0.05
#we need the alpha/2 percentile of a N(0,1), z_a
z_a   = sc.norm.ppf(q=1-alpha/2,loc=0,scale=1) 
#the lower and upper bound of the confidence interval are
mu_l  = Xb - z_a * sg/np.sqrt(n)
mu_u  = Xb + z_a * sg/np.sqrt(n)
CI_mu = [mu_l,mu_u]
print(CI_mu)

###############################################################################
# Example 2, analysis of Xbar and S2
###############################################################################
#we simulate a sample of N(mu,sg) with parameters mu and sg 
mu= 100
sg= 5
n = 100
X = sc.norm.rvs(size=n,loc=mu, scale=sg) 
#we calculate the mean X^bar and the empirical standard deviation S
Xb = np.mean(X)  
S  = np.std(X)
print([Xb,S])

###############################################################################
# Imagine that we do not know sigma^2 but want estimate the probability that
# the spread between S^2 and sigma^2 is bigger than 5 P(|S^2-sigma^2|>5)
# sigma is assumed to be known
###############################################################################
c1 =  (n-1)*(1-5/sg**2)
c2 =  (n-1)*(5/sg**2+1)
Proba = sc.chi2.cdf(c1,df=n-1)+(1-sc.chi2.cdf(c2,df=n-1))
print(Proba)
###############################################################################
# Imagine that we do not know sigma^2 but want a confidence interval with
# alpha = 5%.
###############################################################################
alpha = 0.05
#we need the alpha/2 and 1-alpha/2 percentile of a chi-square r.v., z_a
c_l   = sc.chi2.ppf(q=1-alpha/2,df=n-1)
c_u   = sc.chi2.ppf(q=alpha/2,df=n-1) 
 
sg2_l = S**2*(n-1)/c_l
sg2_u = S**2*(n-1)/c_u
CI_sg = [np.sqrt(sg2_l),np.sqrt(sg2_u)]
print(CI_sg)


###############################################################################
# Example 3, analysis of Xbar and S2
###############################################################################
#we simulate a sample of N(mu,sg) with parameters mu and sg 
mu= 100
sg= 10
n = 100
X = sc.norm.rvs(size=n,loc=mu, scale=sg) 
#we calculate the mean X^bar and the empirical standard deviation S
Xb = np.mean(X)  
S  = np.std(X)
print([Xb,S])
###############################################################################
# We do not know mu and sigma^2 but want a confidence interval 
# for mu and alpha = 5%.
###############################################################################
alpha = 0.05
#we need the alpha/2  percentile of a t r.v., z_a
t_a   = sc.t.ppf(q=1-alpha/2,df=n-1)

mu_l  = Xb - t_a * S/np.sqrt(n)
mu_u  = Xb + t_a * S/np.sqrt(n)
CI_mu = [mu_l,mu_u]
print(CI_mu)