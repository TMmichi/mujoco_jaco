from numpy.random import normal



class OUNoise:
    """
    OUNoise(dt, theta, sigma)

    Discrete-time implementation of the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    Parameters
    ----------
    dt : float
        Step time duration for discrete-time OU process.
    mu : float
        Mean value of OU process.
    theta : float
        Parameter for OU process.
    sigma : float
        Parameter for OU process.
    x_init : float
        Initial state of OU process.
    """

    def __init__(self, dt, theta, sigma, mu=0, x_init=0):
        
        self.reset(x_init=x_init)
        self._dt    = dt
        self._mu    = mu
        self._theta = theta
        self._sigma = sigma


    def reset(self, x_init=0):

        self._x_init = x_init
        self._x = self._x_init


    def evolve(self):

        x = self._x
        dx = self._theta * (self._mu - x) * self._dt + self._sigma * normal()
        self._x = x + dx

        return self._x

if __name__=="__main__":

    import matplotlib.pyplot as plt
    from numpy import arange
    dt = 0.1
    ou = OUNoise(dt, 0.1, 0.1, x_init=3)

    t_hist = list()
    x_hist = list()
    for t in arange(0, 10, dt):
        x = ou.evolve()
        x_hist.append(x)
        t_hist.append(t)

    plt.plot(t_hist, x_hist)
    plt.show()