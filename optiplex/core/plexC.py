from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                 con: Callable = lambda v: np.zeros(0),
                 scale: float = 1.0,
                 mu: np.ndarray = None,
                 ):

        self.subproblems = subproblems
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.time = None
        self.con = con
        self.y = np.zeros_like(con(self.x)) # Lagrange multipliers
        self.d = len(self.y) # number of constraints
        self.history = [self.x.copy()] # data dictionary/list
        self.x_time = [0.0] # time history for each x update
        self.scale = scale
        self.mu = np.ones(self.d) if mu is None else mu # augmented Lagrangian penalty parameter
        self.mu_history = [self.mu]

    def solve(self, 
              max_outer_iter: int=1000, # maximum number of outer iterations
              max_inner_iter: int=1000, # maximum number of inner iterations
              rho: float=1.2, # penalty increase factor
              eps: float=1e-1,
              tol: float=1e-6,
              max_mu: float = 1000.0,
              tau: float = 0.5, # factor for increasing mu based on constraint violation
              ) -> None:
        
        assert rho > 1
        t1 = time.perf_counter()

        c_prev = self.con(self.x)
        # feas_prev = np.linalg.norm(c_prev)

        for k in range(max_outer_iter):

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 

                    self.x = subP(self.x, self.y, self.mu)
                    self.history.append(self.x.copy())
                    self.mu_history.append(self.mu)
                    self.x_time.append(time.perf_counter() - t1)

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = z_new - z_old
                relative_step = np.linalg.norm(step * self.scale)

                # Print inner iteration data
                print(f"pr_itr={j:03d} | "f"rel_stp={relative_step:.3e} | ")

                if relative_step <= eps:
                    print('-Primal loop converged!-')
                    break

            # Evaluate the constraints
            c_new = self.con(self.x)
            feas = np.linalg.norm(c_new, order='inf')

            if feas <= tol:
                print('-Dual loop converged with feasibility: ', feas)
                break

            # self.y += mu * c # Always update the multipliers
            self.y += self.mu * self.mu * c_new
            # L = f(x) + y^T c(x) + 0.5 * ||mu * c(x)||^2

            # update mu on a per-constraint basis
            for i in range(self.d):
                c_new_i = c_new[i]
                c_prev_i = c_prev[i]

                if abs(c_new_i) > tau * abs(c_prev_i):
                    print(f'increasing mu for constraint {i}')
                    self.mu[i] = min(rho * self.mu[i], max_mu)


            print(f"du_itr={k:03d} | "
                  f"feas={feas:.3e} | "
                  f"mu={self.mu:.2f} | "
                  f"y={np.linalg.norm(self.y):.3e}"
                  )


        self.time = time.perf_counter() - t1

        return None