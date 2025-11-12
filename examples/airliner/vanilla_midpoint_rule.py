import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np



def midpoint_rule(f, t0, y0, h, n, args):
    """
    Implement the second-order Runge–Kutta (Midpoint) method.

    Parameters:
        - f: Function representing the differential equation y' = f(t, y)
        - t0: Initial value of t
        - y0: Initial value of y(t0)
        - h: Step size
        - n: Number of steps to take
        - args: Additional arguments passed to f
    """

    solution = [y0]
    t = t0
    Y = y0

    for _ in range(n):
        k1 = h * f(t, Y, args)
        k2 = h * f(t + 0.5 * h, Y + 0.5 * k1, args)

        Y = Y + k2
        t = t + h

        solution.append(Y)

    return jnp.array(solution)




def jax_midpoint(f, t0, y0, h, n, args):
    """
    Accelerated (JAX-compatible) Midpoint Rule integrator using jax.lax.scan.

    Parameters:
        - f: Function representing the differential equation y' = f(t, y)
        - t0: Initial value of t
        - y0: Initial value of y(t0)
        - h: Step size
        - n: Number of steps to take
        - args: Additional arguments passed to f
    """

    y0 = jnp.atleast_1d(y0)

    solution = jnp.zeros((n + 1, len(y0)))
    solution = solution.at[0, :].set(y0)

    def step(carry, _):
        t, Y = carry
        # t_half = t + 0.5 * h

        k1 = h * f(t, Y, args)
        k2 = h * f(t + 0.5 * h, Y + 0.5 * k1, args)

        Y_next = Y + k2
        t_next = t + h

        return (t_next, Y_next), Y_next

    (_, _), ys = jax.lax.scan(step, (t0, y0), None, length=n)

    return solution.at[1:, :].set(ys)





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    def growth_rate(t, y, args):
        return args[0] * y

    t0 = 0.0
    y0 = 100.0
    h = 0.1
    n = 10000
    k = 0.05

    t1 = time.time()

    solution = midpoint_rule(growth_rate, t0=t0, y0=y0, h=h, n=n, args=[k])
    # solution = jax_midpoint(growth_rate, t0=t0, y0=y0, h=h, n=n, args=[k])

    t2 = time.time()
    print('time: ', t2 - t1)

    t_values = np.linspace(t0, t0 + n * h, n + 1)
    P_values = np.array(solution)
    
    plt.plot(t_values, P_values, label='Population Growth')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Growth Over Time')
    plt.legend()
    plt.show()