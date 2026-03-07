import numpy as np
import jax.numpy as jnp
import jax

N = 5
x = np.random.rand(N)

# Test with all variables, which should be rank deficient

def consensus(x):
    x_bar = jnp.mean(x)
    return x - x_bar

jac = jax.jacobian(consensus)(x)

print(jac)

rank = np.linalg.matrix_rank(jac)
print("Rank of the Jacobian: ", rank)


# Now test with one less variable, which should be full rank

def consensus(x):
    x_bar = jnp.mean(x)
    return x[:-1] - x_bar


jac = jax.jacobian(consensus)(x)

print(jac)

rank = np.linalg.matrix_rank(jac)
print("Rank of the Jacobian: ", rank)