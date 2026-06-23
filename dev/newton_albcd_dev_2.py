import numpy as np
import modopt as mo
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from typing import List, Callable
import time
from itertools import combinations

# v_init = [1.0, -1.0, 1.0, -1.0]
# v_init = [-1.0, 1.0, -1.0, 1.0]
v_init = [-0.5, 1.0, -0.5, 1.0]

x1_1_history = [v_init[0]]
x2_1_history = [v_init[1]]
x1_2_history = [v_init[2]]
x2_2_history = [v_init[3]]

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
    num_blocks = len(variables)
    indices = list(range(num_blocks))
    pairs = list(combinations(indices, 2))
    
    # remove an arbitrary pair so the constraints are linearly independent
    # (e.g. remove the last pair)
    if len(pairs) > 1:
        print('removing one pair')
        pairs = pairs[:-1]

    c = jnp.array([variables[i] - variables[j] for i, j in pairs])

    return c.flatten()


def subproblem1(v_init, y, mu):
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    v0 = np.concatenate([np.atleast_1d(x1_1), np.atleast_1d(x2_1)])

    def jax_obj(v):
        x1_1 = v[0]
        x2_1 = v[1]
        obj = jnp.squeeze(x1_1**2 + x2_1**2 - 1.5 * x1_1 * x2_1)

        c_1 = combo([x1_1, x1_2])
        c_2 = combo([x2_1, x2_2])
        c = jnp.concatenate([c_1, c_2])

        return obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    def jax_con(v):
        x1_1, x2_1 = v[0], v[1]
        con = x1_1**2 + x2_1**2
        return con.flatten()
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0.5**2, cu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x']

    x1_1_history.append(ans[0])
    x2_1_history.append(ans[1])
    x1_2_history.append(x1_2)
    x2_2_history.append(x2_2)

    return [ans[0], ans[1], x1_2, x2_2]


def subproblem2(v_init, y, mu):
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    v0 = np.concatenate([np.atleast_1d(x1_2), np.atleast_1d(x2_2)])

    def jax_obj(v):
        x1_2 = v[0]
        x2_2 = v[1]
        obj = jnp.squeeze(x1_2**2 + x2_2**2 - 1.5 * x1_2 * x2_2)

        c_1 = combo([x1_1, x1_2])
        c_2 = combo([x2_1, x2_2])
        c = jnp.concatenate([c_1, c_2])

        return obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    def jax_con(v):
        x1_2, x2_2 = v[0], v[1]
        con = x1_2**2 + x2_2**2
        return con.flatten()
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0.5**2, cu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x']

    x1_1_history.append(x1_1)
    x2_1_history.append(x2_1)
    x1_2_history.append(ans[0])
    x2_2_history.append(ans[1])

    return [x1_1, x2_1, ans[0], ans[1]]


def con(v_init):
    
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    c_1 = combo([x1_1, x1_2])
    c_2 = combo([x2_1, x2_2])
    return jnp.concatenate([c_1, c_2])



def AL(z, y, mu):

    x1_1, x2_1, x1_2, x2_2 = z

    f = (
        x1_1**2 + x2_1**2 - 1.5*x1_1*x2_1
        + x1_2**2 + x2_2**2 - 1.5*x1_2*x2_2
    )

    c1 = combo([x1_1, x1_2])
    c2 = combo([x2_1, x2_2])

    c = jnp.concatenate([c1, c2])

    return f + y @ c + 0.5 * mu * jnp.sum(c**2)





class PlexC():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                 con: Callable = lambda v: np.zeros(0),
                 mu: np.ndarray | float = 1.0, # positive penalty parameter(s)
                 max_mu: float = 1e3, # maximum penalty parameter
                 rho: float = 1.2, # penalty increase factor
                 tau: float = 0.5, # factor for increasing mu based on constraint violation
                 tol: float = 1e-3, # outer loop feasibility tolerance
                 eps: float = 1e-5, # inner loop convergence tolerance
                 max_y: float = 1e6, # maximum Lagrange multiplier value
                 ):

        self.subproblems = subproblems
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.con = con

        self.initial_constraint_values = self.con(self.x)
        self.y = np.zeros_like(self.initial_constraint_values) # Lagrange multipliers
        self.history = [self.x.copy()]
        self.feasibility = [max(abs(self.initial_constraint_values))]
        self.x_time = [0.0]
        self.y_history = [self.y.copy()]

        self.mu = mu # penalty parameter(s)
        assert np.all(self.mu >= 0)
        self.max_mu = max_mu
        # self.mu_history = [self.mu.copy()]
        self.mu_history = [self.mu]
        self.t0 = None
        self.tf = None

        assert rho >= 1
        self.rho = rho
        self.tau = tau
        self.tol = tol
        self.eps = eps
        self.max_y = max_y


    def _pack_x(self):
        return np.asarray(
            np.concatenate(
                [np.atleast_1d(xi).ravel() for xi in self.x]
            ),
            dtype=float,
        )
    

    def _dual_newton_step(self):

        z = jnp.asarray(self._pack_x())
        y = jnp.asarray(self.y)

        # current constraint residual
        c = np.asarray(self.con(self.x), dtype=float)

        # exact augmented-Lagrangian Hessian
        H = np.asarray(
            jax.hessian(
                lambda z_: AL(z_, y, self.mu)
            )(z)
        )

        # exact constraint Jacobian
        J = np.asarray(
            jax.jacobian(
                lambda z_: jnp.asarray(con(z_))
            )(z)
        )

        #
        # dual Hessian
        #
        # ∇²d = -J H^{-1} Jᵀ
        #
        Hinv_JT = np.linalg.solve(H, J.T)

        Hd = -(J @ Hinv_JT)

        #
        # Newton system
        #
        # Hd Δy = -c
        #
        reg = 1e-10*np.eye(Hd.shape[0])

        try:
            dy = np.linalg.solve(Hd - reg, -c)
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(Hd - reg, -c, rcond=None)[0]

        return dy


    def _update_mu(self, c_new, c_old) -> None:

        if isinstance(self.mu, np.ndarray):

            num = 0
            for i, (f_new, f_old) in enumerate(zip(abs(c_new), abs(c_old))):
                if f_new > self.tau * f_old and f_new > self.tol:
                    self.mu[i] = min(self.rho * self.mu[i], self.max_mu)
                    num += 1

        elif isinstance(self.mu, (float, int)):

            if max(abs(c_new)) > self.tau * max(abs(c_old)) and max(abs(c_new)) > self.tol:
                self.mu = min(self.rho * self.mu, self.max_mu)

        return None


    def _inner_loop(self) -> None:
        
        for subP in self.subproblems: 

            self.x = subP(self.x, self.y, self.mu)
            self.history.append(self.x.copy())
            # self.mu_history.append(self.mu.copy())
            self.mu_history.append(self.mu)
            self.x_time.append(time.perf_counter() - self.t0)


    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=10,  # maximum number of inner iterations
              ) -> None:
        
        self.t0 = time.perf_counter()

        for k in range(max_outer_iter):

            c_old = self.con(self.x)

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                # block coordinate descent inner loop
                self._inner_loop()

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = abs(z_new - z_old)
                denom = np.maximum(abs(z_old), abs(z_new))
                denom = np.maximum(denom, 1e-5) # floor
                rel_step = max(step / denom)

                print(f"pr_itr={j:03d} | "f"rel_stp={rel_step:.3e} | ")

                if rel_step <= self.eps:
                    print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break


            c_new = self.con(self.x)
            # print(abs(c_new))
            feas = np.max(abs(c_new))
            self.feasibility.append(feas)

            if feas <= self.tol:
                print('-Dual loop converged with feasibility: ', feas, ' in ', k, ' dual iterations!-')
                break

            # self.y += self.mu * c_new # always update the multipliers
            # self.y += np.diag(self.mu) @ c_new # always update the multipliers

            # if isinstance(self.mu, np.ndarray):
            #     self.y += np.diag(self.mu) @ c_new
            # if isinstance(self.mu, (float, int)):
            #     self.y += self.mu * c_new

            dy = self._dual_newton_step()

            self.y += dy
            self.y = np.clip(self.y,
                            -self.max_y,
                            self.max_y)

            self.y_history.append(self.y)

            print(
                f"||c||={np.linalg.norm(c_new):.3e} | "
                f"||dy||={np.linalg.norm(dy):.3e}"
            )


            self._update_mu(c_new, c_old)

            c_old = c_new

            print(f"du_itr={k:03d} | "
                  f"feas={feas:.3e} | "
                  f"max mu={np.max(self.mu):.3e} | "
                  f"min mu={np.min(self.mu):.3e} | "
                  f"y={np.linalg.norm(self.y):.3e} | "
                  )

        self.tf = time.perf_counter() - self.t0

        return None













opt = PlexC(subproblems=[subproblem1, subproblem2],
            x_init=v_init,
            con=con,
            mu=10,
            max_mu=1e6,
            rho=1.5,
            tau=0.5,
            tol=1e-7, # outer loop feasibility
            eps=1e-5, # inner loop convergence
            )

opt.solve(max_outer_iter=100, 
          max_inner_iter=100,
          )

print('Solution: ', opt.x)
print('Time (s): ', opt.tf)



plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(x1_1_history, x2_2_history, 's-', color='tab:orange', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_1, y_2)$')
plt.plot(x1_2_history, x2_1_history, 'o-', color='tab:purple', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_2, y_1)$')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

theta = np.linspace(0, 2*np.pi, 100)
circle_x = 0.5 * np.cos(theta)
circle_y = 0.5 * np.sin(theta)
plt.plot(circle_x, circle_y, '--', color='black', linewidth=2, alpha=0.5)
plt.fill(circle_x, circle_y, color='black', alpha=0.3)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend()
plt.gca().set_aspect('equal')

# plt.savefig('augmented_lagrangian_circle_constraint.pdf', bbox_inches='tight')
plt.show()