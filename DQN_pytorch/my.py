import numpy as np
np.random.seed(1)
y = lambda x: x**2

mu, sigma = 5, 5
n = 100
for i in range(10):
    z = np.random.normal(mu, sigma, n)
    y_z = y(z)
    z_selected = z[np.argsort(y_z)[:10]]
    mu,sigma = z_selected.mean(),z_selected.std()
    print("iteration {0} mu: {1} sigma: {21}".format(i,mu,sigma))