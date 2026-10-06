# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:39:32 2020

@author: dasha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import statsmodels.api as sm
#statsmodels.stats.weightstats.DescrStats
import os

# we check the path and open the data set
print(os.getcwd())
data = pd.read_csv("spring.csv",sep=';' ) 
# we have a dataframe, we convert it in an array and just
datn = data.values
n    = datn.shape[0]
#####################################################################
# X explanatory variable : the mass in kilogram
# Y response variable : the deflection in meter
# we calculate beta 1 and beta 0
#####################################################################
X = datn[:,0] ;   Y = datn[:,1]
Xb=np.mean(X) ;   Yb= np.mean(Y)
b1=np.sum((X-Xb)*(Y-Yb))/np.sum((X-Xb)**2)   
b0=Yb-b1*Xb
#####################################################################
# But scipy also proposes a function for linear regressions :
#####################################################################
slope, intercept, r_value, p_value, std_err = sc.linregress(X,Y)
print(np.round([b0,intercept,b1,slope],3))


# Plot of measures versus regressions
x  = np.linspace(min(X),  max(X), 100)
y  = intercept + slope*x
plt.plot(X, Y,'x',c='r')
plt.plot(x,y,c='b')
plt.xlabel('Mass')
plt.ylabel('Deflection')
plt.title('Hooke Law')
plt.grid(True)
plt.legend(['Observation','Regression'])

#####################################################################
# Best package for linear regression library statsmodel
# We need to add a constant explanatory variable (for beta0) 
# this constant is not needed in scipy
Xm       = sm.add_constant(X)
# we fit by least square minimization the model
results = sm.OLS(Y,Xm).fit()
# results is an object with several properties
print(results.summary())

#you can calculate the t stat yourself
Sxx   = sum((X-Xb)**2)                    #Sxx       
X2b   = np.mean(X**2)                     #mean of X^2
Yhat  = results.predict(Xm)               #prediction
sghat = np.sqrt(sum((Y-Yhat)**2)/(n-2))   #estimate of sigma
# Test 1: beta1 =0 v.s. beta1<>0
T1    = b1/(sghat*np.sqrt(Sxx**-1))      #test statistics
pval1 = 2*(1-sc.t.cdf(abs(T1),df=n-2))   #p-value
#confidence interval for beta1
CI1   = b1+[-sghat*np.sqrt(Sxx**-1)*sc.t.ppf(q=0.975,df=n-2),  \
        +sghat*np.sqrt(Sxx**-1)*sc.t.ppf(q=0.975,df=n-2) ]
#very low pvalue , we reject H0 : b1=0 at 5%
# Test 2: beta0 =0 v.s. beta0<>0
T0   = b0/(sghat*np.sqrt(X2b*Sxx**-1))  #test statistics
pval0= 2*(1-sc.t.cdf(abs(T0),df=n-2))    #p-value
#confidence interval for beta0
CI0   = b0+[-sghat*np.sqrt(X2b*Sxx**-1)*sc.t.ppf(q=0.975,df=n-2),  \
           +sghat*np.sqrt(X2b*Sxx**-1)*sc.t.ppf(q=0.975,df=n-2) ]
#low pvalue , we reject H0 : b0=0 at 5% but at 7% we accept...
#you can check that you get the same  p-values as OLS
print(np.round(results.pvalues,4))
print(np.round([pval0,pval1],4))

#####################################################################
# the model can be used for prediction
#####################################################################
X_prime = np.linspace(1, 2, 10)
X_prime = sm.add_constant(X_prime) # add constant as we did before
# Now we calculate the predicted values
y_hat = results.predict(X_prime)

# we plot the observations and predictions
plt.scatter(Xm[:,1], Y, alpha=0.3) # Plot the raw data
plt.title('Hooke Law')
plt.xlabel("Mass")
plt.ylabel("Deflection")
plt.plot(X_prime[:, 1], y_hat, 'r', alpha=0.9) # Add the regression line, colored in red
plt.grid(True)
plt.legend(['Observation','Prediction'])
