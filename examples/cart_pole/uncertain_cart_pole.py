import numpy as np
import jax.numpy as jnp
from modopt import CSDLAlphaProblem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
from scipy.stats.qmc import LatinHypercube, scale

# sample the space of uncertain parameters using Latin hypercube sampling
# uncertain parameters: g, mu_cart, mu_pole
g_u = 9.81 + 0.05
g_l = 9.81 - 0.05
mu_cart_u = 0.03 + 0.01
mu_cart_l = 0.03 - 0.01
mu_pole_u = 0.03 + 0.01
mu_pole_l = 0.03 - 0.01
N = 2
sampler = LatinHypercube(d=3, seed=0)
samples = scale(sampler.random(N), 
                l_bounds=[g_l, mu_cart_l, mu_pole_l], 
                u_bounds=[g_u, mu_cart_u, mu_pole_u])

print(samples)

# parameters
n = 30
dt = 2 / n
mc = 2
d = 0.8




def jax_obj(v):

    ji = []
    for i in range(N):
        ui = 
        ji = 0.5 * dt * jnp.sum(ui[:-1]**2 + ui[1:]**2)
    ji = jnp.array(ji)

    return jnp.sum(ji) / N