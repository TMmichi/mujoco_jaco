def alpha(m,v):
    return m*((m*(1-m)/v)-1)

def beta(m,v):
    return (1-m)*((m*(1-m)/v)-1)

m = 0.85

v = 0.01

for i in range(500):
    print(m, alpha(m,v), beta(m,v))
    m += 0.0001