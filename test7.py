import numpy as np
import time

now = time.time()
for i in range(10000):
    ind = int(np.random.rand() < 0.5)
print('rand: ', time.time()-now)
now = time.time()
for i in range(10000):
    np.random.randint(0,2)
print('randint: ', time.time()-now)