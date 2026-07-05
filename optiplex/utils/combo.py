from itertools import combinations
import jax.numpy as jnp




def combo(variables: list) -> jnp.ndarray:
    """
    Compute the flattened vector of pairwise differences between input variables.

    Given a sequence of arrays (blocks) [x0, x1, ..., x_{N-1}], this function
    computes all pairwise differences x_i - x_j for i < j, in the order
    produced by itertools.combinations(range(N), 2). The per-pair results are
    stacked into a JAX array and the final result is returned as a 1-D array
    (flattened).

    Parameters
    ----------
    variables : list
            Sequence of JAX arrays (e.g., `jnp.ndarray`). All arrays should have the
            same shape; the function does not perform automatic broadcasting. If the
            arrays have different shapes, the underlying JAX operations may raise
            an error.

    Returns
    -------
    jnp.ndarray
            1-D JAX array containing the flattened pairwise differences x_i - x_j
            for all pairs (i, j) with i < j. If `len(variables) < 2`, an empty
            1-D array is returned.
    """

    # # ensure that every variable is the same shape
    # if not all(var.shape == variables[0].shape for var in variables):
    #     raise ValueError("consensus variables must have the same shape.")


    num_blocks = len(variables)
    indices = list(range(num_blocks))
    pairs = list(combinations(indices, 2))

    
    # remove an arbitrary pair so the constraints are linearly independent
    # (e.g. remove the last pair)
    if len(pairs) > 1:
        # print('removing one pair')
        pairs = pairs[:-1]
    # print(pairs)
    


    c = jnp.array([variables[i] - variables[j] for i, j in pairs])

    return c.flatten()




if __name__ == "__main__":

    # example usage
    # define some example variables
    l1 = jnp.array([1.0])
    l2 = jnp.array([2.0])
    l3 = jnp.array([3.0])

    c = combo([l1, l2, l3])
    print(c)