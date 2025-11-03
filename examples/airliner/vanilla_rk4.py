import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

# ordinary version
def runge_kutta_4(f, t0, y0, h, n, args):
    """
    Implement the fourth-order Runge-Kutta method.
 
    Parameters:
        - f: Function representing the differential equation y' = f(t, y)
        - t0: Initial value of t
        - y0: Initial value of y(t0)
        - h: Step size
        - n: Number of steps to take
    """

    solution = [y0]
    t = t0
    Y = y0
 
    for _ in range(n):
        k1 = h * f(t, Y, args)
        k2 = h * f(t + 0.5 * h, Y + 0.5 * k1, args)
        k3 = h * f(t + 0.5 * h, Y + 0.5 * k2, args)
        k4 = h * f(t + h, Y + k3, args)
 
        Y = Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h

        solution.append(Y)
 
    return jnp.array(solution)



# optimized jax version
def jax_rk4(f, t0, y0, h, n, args):

    y0 = jnp.atleast_1d(y0)

    solution = jnp.zeros((n + 1, len(y0)))
    # prepend initial condition
    solution = solution.at[0, :].set(y0)

    def step(carry, _):
        t, Y = carry
        k1 = h * f(t, Y, args)
        t_plus_half_h = t + 0.5 * h
        t_plus_h = t + h
        # k2 = h * f(t + 0.5 * h, Y + 0.5 * k1, args)
        # k3 = h * f(t + 0.5 * h, Y + 0.5 * k2, args)
        k2 = h * f(t_plus_half_h, Y + 0.5 * k1, args)
        k3 = h * f(t_plus_half_h, Y + 0.5 * k2, args)
        # k4 = h * f(t + h, Y + k3, args)
        k4 = h * f(t_plus_h, Y + k3, args)

        Y_next = Y + (k1 + 2*k2 + 2*k3 + k4) / 6
        # t_next = t_plus_h

        # return (t_next, Y_next), Y_next
        return (t_plus_h, Y_next), Y_next

    # scan for n steps
    (_, _), ys = jax.lax.scan(step, (t0, y0), None, length=n)

    # prepend initial condition
    return solution.at[1:, :].set(ys)




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def growth_rate(t, y, args):
        return args[0] * y

    t0 = 0.0
    y0 = 100.0
    h = 0.1
    n = 100
    k = 0.05

    # solution = runge_kutta_4(growth_rate, t0=t0, y0=y0, h=h, n=n, args=[k])
    solution = jax_rk4(growth_rate, t0=t0, y0=y0, h=h, n=n, args=[k])

    t_values = np.linspace(t0, t0 + n * h, n + 1)
    P_values = np.array(solution)
    
    plt.plot(t_values, P_values, label='Population Growth')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Growth Over Time')
    plt.legend()
    plt.show()