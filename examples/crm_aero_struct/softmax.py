# import jax.numpy as jnp
# import jax
# import matplotlib.pyplot as plt


# random_vector = jnp.array([1.0, 2.0, 3.0])
# softmax_vector = jax.nn.softmax(random_vector)
# print("Random vector:", random_vector)
# print("Softmax of the random vector:", softmax_vector)

import jax
import jax.numpy as jnp


# def ks_max(x: jnp.ndarray, rho: float = 100.0) -> jnp.ndarray:
#     """
#     Smooth KS-max (Kreisselmeier-Steinhauser) approximation of the maximum.

#     KS_max(x, rho) = (1/rho) · ln sum exp(rho · x_i)

#     Uses the log-sum-exp trick for numerical stability:
#       = x_max + (1/rho) · ln sum exp(rho · (x_i - x_max))

#     Args:
#         x:   Input array.
#         rho: Smoothing parameter - larger implies closer to true max. Default 100.

#     Returns:
#         Scalar smooth approximation of max(x).
#     """
#     x_max = jnp.max(x)
#     return x_max + jnp.log(jnp.sum(jnp.exp(rho * (x - x_max)))) / rho

def ks_max(x: jnp.ndarray, rho: float = 100.0) -> jnp.ndarray:
    """
    Smooth KS-max (Kreisselmeier-Steinhauser) approximation of the maximum.

    KS_max(x, rho) = (1/rho) · ln sum exp(rho · x_i)
    """
    return jnp.log(jnp.sum(jnp.exp(rho * (x)))) / rho



if __name__ == "__main__":
    # key = jax.random.PRNGKey(42)
    # x = jax.random.normal(key, shape=(8,))

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 500, 800, 100])

    print("x              :", x)
    print("true max       :", jnp.max(x))
    print()
    for rho in [1e-2, 1.0, 10.0, 50.0, 100.0]:
        print(f"ks_max(rho={rho:5.0f}) : {ks_max(x, rho):.6f}")

    # # JAX auto-diff smooth max is fully differentiable
    # print("\n── Gradient (rho=50) ──")
    # grad_fn = jax.grad(lambda x: ks_max(x, rho=50.0))
    # print("grad :", grad_fn(x))