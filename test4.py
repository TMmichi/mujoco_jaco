import numpy as np
from scipy.special import gamma, digamma

def alpha(m,v):
    return m*((m*(1-m)/v)-1)

def beta(m,v):
    return (1-m)*((m*(1-m)/v)-1)

def bcon(a,b):
    return gamma(a)*gamma(b)/gamma(a+b)

def phi(a):
    return digamma(a)

def entropy(a,b):
    return np.log(bcon(a,b)) - (a-1)*phi(a) - (b-1)*phi(b) + (a+b-2)*phi(a+b)

m = 0.5

v = 0.01

for i in range(500):
    print(v, entropy(alpha(m,v),beta(m,v)))
    v += 0.0001