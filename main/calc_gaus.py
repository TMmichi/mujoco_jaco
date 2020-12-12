import numpy as np

def gaus(mean, std, a):
    val1 = (((mean-a)/std)**2)*0.5 + 0.5*np.log(2.0*np.pi) + np.log(std)
    val2 = np.log(1-(a)**2)
    return val1+val2

# print(np.log(1-(-0.999927938)**2))
print(gaus(-0.99937886, 0.953573108, -0.999927938))


# -59.8994064
# -0.190275803 -0.999927938 -0.993390799 -0.994242609 -0.601499677 0.989209116 -0.99999 -0.997575939
# 0.56807065 -0.99937886 -0.998498857 -0.983738422 -0.973246157 0.993206 -0.999989808 -0.997065306]
# 0.921017289 0.953573108 0.767502069 0.986610055 0.918251872 0.915425301 0.670900583 0.407740474