import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
jax.config.update("jax_enable_x64", True)
import jaxopt


# trivial numpy version of backward Euler with Newton-Raphson
def backward_euler(f, t0, y0, h, n, args=(), jac=None,
                   tol=1e-8, max_iter=50):

    y0 = np.array(y0, dtype=float)
    dim = y0.size if y0.ndim > 0 else 1

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, dim))

    t[0] = t0
    y[0] = y0

    for n in range(n):
        t_next = t[n] + h
        y_guess = y[n].copy()

        # Newton-Raphson iteration
        for _ in range(max_iter):

            # Residual: G(y) = y - y_n - h f(t_{n+1}, y)
            G = y_guess - y[n] - h * np.atleast_1d(
                f(t_next, y_guess, args)
            )

            if np.linalg.norm(G) < tol: break

            # Jacobian of G: I - h * df/dy
            J = jac(t_next, y_guess, args)

            G_jac = np.eye(dim) - h * J

            delta = np.linalg.solve(G_jac, -G)
            y_guess = y_guess + delta

            if np.linalg.norm(delta) < tol:
                break
        else:
            raise RuntimeError(f"Newton failed to converge at step {n}")

        y[n + 1] = y_guess
        t[n + 1] = t_next

    return y




# JAX version of backward Euler using Newton-Raphson with a fixed number of iterations for autodiff
def jax_backward_euler_fixed(f, t0, y0, h, n, args=(),
                   tol=1e-8, max_iter=20):

    y0 = jnp.atleast_1d(y0)
    dim = y0.shape[0]

    def newton_solver(y_prev, t_next):

        def body_fun(i, state):
            y_guess, converged = state

            G = y_guess - y_prev - h * f(t_next, y_guess, args)

            J_f = jax.jacobian(lambda y: f(t_next, y, args))(y_guess)
            G_jac = jnp.eye(dim) - h * J_f

            delta = jnp.linalg.solve(G_jac, -G)
            y_new = y_guess + delta

            # convergence check
            res_norm = jnp.linalg.norm(G)
            step_norm = jnp.linalg.norm(delta)
            is_converged = jnp.logical_or(
                res_norm < tol,
                step_norm < tol
            )

            # freeze updates after convergence
            y_next = jnp.where(
                converged[..., None],
                y_guess,
                y_new
            )

            return (y_next, jnp.logical_or(converged, is_converged))

        y_init = y_prev
        state_init = (y_init, False)

        y_final, _ = lax.fori_loop(
            0, max_iter,
            body_fun,
            state_init
        )

        return y_final

    def step(carry, _):
        t_prev, y_prev = carry
        t_next = t_prev + h
        y_next = newton_solver(y_prev, t_next)
        return (t_next, y_next), y_next

    (_, _), ys = lax.scan(
        step,
        (t0, y0),
        None,
        length=n
    )

    ys = jnp.vstack([y0, ys])
    return ys



def newton_fpi(G, y_init, data, tol=1e-9, maxiter=100):

    def T(y, data): # Define the Newton fixed point map
        res = G(y, data) # Compute residual
        J = jax.jacrev(G)(y, data) # Compute Jacobian
        delta = jnp.linalg.solve(J, -res) # Newton step
        return y + delta

    fpi = jaxopt.FixedPointIteration(fixed_point_fun=T,
                                     maxiter=maxiter,
                                     tol=tol,
                                     )

    return fpi.run(y_init, data).params


def backward_euler_step(y_prev, t_next, h, args, i, f):

    def G(y_next, data): # Residual function with all differentiable values explicit
        # y_prev, t_next, h, args = data
        y_prev, t_next, h, args, i = data
        return y_next - y_prev - h * f(t_next, y_next, args, i)

    y_init = y_prev

    data = (y_prev, t_next, h, args, i)
    y_next = newton_fpi(G, y_init, data)
    return y_next

def backward_euler_newton(f, t0, y0, h, n, args):

    y0 = jnp.atleast_1d(y0)

    def step(carry, _):
        t_prev, y_prev, i = carry
        t_next = t_prev + h
        y_next = backward_euler_step(y_prev, t_next, h, args, i, f)
        return (t_next, y_next, i + 1), y_next

    (_, _, _), ys = lax.scan(step, (t0, y0, 0), None, length=n)

    return jnp.vstack([y0, ys])




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example: y' = -lambda * y
    def f(t, y, args):
        lam, = args
        return -lam * y

    def jac(t, y, args):
        lam, = args
        return np.array([[-lam]])
    

    n = 100
    h = 0.01
    t = np.linspace(0, n*h, n+1)

    y = backward_euler(f, t0=0.0, y0=[1.0],
                        h=h, n=n,
                        args=(10.0,), jac=jac)


    plt.plot(t, y, label='Backward Euler')
    plt.plot(t, np.exp(-10.0 * t), label='Exact Solution', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()

    # y = jax_backward_euler(f, t0=0.0, y0=[1.0],
    #                     h=h, n=n,
    #                     args=(10.0,))
    y = jax_backward_euler_fixed(f, t0=0.0, y0=[1.0],
                        h=h, n=n,
                        args=(10.0,))


    plt.plot(t, y, label='Jax Backward Euler')
    plt.plot(t, np.exp(-10.0 * t), label='Exact Solution', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()


    # y = backward_euler_broyden(f, t0=0.0, y0=[1.0],
    #                     h=h, n=n,
    #                     args=(10.0,))
    y = backward_euler_newton(f, t0=0.0, y0=[1.0],
                        h=h, n=n,
                        args=(10.0,))

    plt.plot(t, y, label='Jax Backward Euler')
    plt.plot(t, np.exp(-10.0 * t), label='Exact Solution', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()