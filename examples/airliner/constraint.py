import numpy as np
import jax.numpy as jnp
from typing import List
from optiplex import combo

def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:
    
    b1 = x_init[3]
    b2 = x_init[7]


    c_l = combo([b1, b2]) # need to expand for changing N
    # return jnp.concatenate((c_l, c_mp))
    return c_l