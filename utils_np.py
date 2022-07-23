import numpy as np
import matplotlib.pyplot as plt
import math

def g_of_sigma(sigma, res=100, n_sample=900000):
    xs = np.linspace(-1, 1, res)
    g = np.zeros(res)
    for i, x in enumerate(xs):
        sample = np.random.randn(n_sample, 2)
        prods = sigma(sample[:, 0]) * sigma(x * sample[:, 0] + math.sqrt(1 - x**2) * sample[:, 1])
        g[i] = np.mean(prods)

    return xs, g

def relu(x):
    y = np.zeros_like(x)
    y[x>0] = x[x>0]
    return(y)

def squared(x):
    return(x**2)

def cubed(x):
    return(x**3)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# poly = np.polynomial.Polynomial([0, 2, -1])
# x = np.linspace(-10, 10, 100)
# plt.figure()
# plt.plot(x, poly(x))
# plt.show()
#
# xs, g = g_of_sigma(poly)
# plt.figure()
# plt.plot(xs, g)
# plt.show()
