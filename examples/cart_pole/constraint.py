import numpy as np
import jax.numpy as jnp
from typing import List
from optiplex import combo

def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:
    
    l1 = x_init[0] # need to expand for changing N
    l2 = x_init[1] 
    mp1 = x_init[2]
    mp2 = x_init[3]

    c_l = combo([l1, l2]) # need to expand for changing N
    c_mp = combo([mp1, mp2]) # need to expand for changing N
    return jnp.concatenate((c_l, c_mp))