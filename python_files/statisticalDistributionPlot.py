# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 08:57:33 2020

@author: DH
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

###############################################################################
# The next section plot the pdf & cdf of a Normal random variable
###############################################################################
mu = 25
sg = 5
x  = np.linspace(sc.norm.ppf(0.01,loc=mu,scale=sg),
                 sc.norm.ppf(0.99,loc=mu,scale=sg), 100)
fx = sc.norm.pdf(x,loc=mu,scale=sg)
Fx = sc.norm.cdf(x,loc=mu,scale=sg)

plt.subplot(1,2,1)
plt.plot(x, fx,'b-', lw=1, alpha=0.6, label='Normal pdf')
plt.title('Normal pdf')
plt.grid(True)
plt.legend(['$\mu=25 , \sigma=5$'])

plt.subplot(1,2,2)
plt.plot(x, Fx,'r-', lw=1, alpha=0.6, label='Normal cdf')
plt.title('Normal cdf')
plt.grid(True)
plt.legend(['$\mu=25 , \sigma=5$'])

###############################################################################
# The next section plot the pdf & cdf of a chi-square random variable
###############################################################################
df1=10
df2=30
x  = np.linspace(sc.chi2.ppf(0.01,df=df1),
                 sc.chi2.ppf(0.99,df=df2), 100)
fx1 = sc.chi2.pdf(x,df=df1)
fx2 = sc.chi2.pdf(x,df=df2)

plt.plot(x, fx1,'b-', lw=1, alpha=0.6)
plt.plot(x, fx2,'r-', lw=1, alpha=0.6)
plt.title('$\chi^2$ pdf')
plt.grid(True)
lg1 = 'n=' + str(df1)
lg2 = 'n=' + str(df2)
plt.legend([lg1,lg2])

###############################################################################
# The next section plot the pdf & cdf of a Student's T and Fisher-Snedecor r.v.
###############################################################################
df1=1
df2=15
x  = np.linspace(sc.t.ppf(0.01,df=df1),
                 sc.t.ppf(0.99,df=df2), 100)
fx1 = sc.t.pdf(x,df=df1)
fx2 = sc.t.pdf(x,df=df2)

plt.subplot(1,2,1)
plt.plot(x, fx1,'b-', lw=1, alpha=0.6)
plt.plot(x, fx2,'r-', lw=1, alpha=0.6)
plt.title('Students ''T pdf')
plt.grid(True)
lg1 = 'n=' + str(df1)
lg2 = 'n=' + str(df2)
plt.legend([lg1,lg2])

df1=5
df2=10
x  = np.linspace(sc.f.ppf(0.01,dfn=df1,dfd=df2),
                 sc.f.ppf(0.99,dfn=df1,dfd=df2), 100)
fx1 = sc.f.pdf(x,dfn=df1,dfd=df2)
fx2 = sc.f.pdf(x,dfn=df2,dfd=df1)

plt.subplot(1,2,2)
plt.plot(x, fx1,'b-', lw=1, alpha=0.6)
plt.plot(x, fx2,'r-', lw=1, alpha=0.6)
plt.title('Fisher F')
plt.grid(True)
lg1 = 'df=(' + str(df1) +',' + str(df2) + ')'
lg2 = 'df=(' + str(df2) +',' + str(df1) + ')'
plt.legend([lg1,lg2])

