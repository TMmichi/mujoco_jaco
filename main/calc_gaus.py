import numpy as np

def gaus(mean, logstd, a):
    std = np.exp(logstd)
    print(std)
    val1 = (((mean-a)/std)**2)*0.5 + 0.5*np.log(2.0*np.pi) + np.log(std)
    val2 = np.log(1-(a)**2)
    return val1+val2

# print(np.log(1-(-0.999927938)**2))
val = gaus(0.349851102, -3.94686103, 0.356499463)
print(val)
print(np.exp(val))