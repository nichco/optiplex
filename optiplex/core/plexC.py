from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                 con: Callable = lambda v: np.zeros(0),
                 mu: np.ndarray = None, # positive penalty parameter(s)
                 max_mu: float = 1e3, # maximum penalty parameter
                 rho: float = 1.2, # penalty increase factor
                 tau: float = 0.5, # factor for increasing mu based on constraint violation
                 tol: float = 1e-3, # outer loop feasibility tolerance
                 eps: float = 1e-3, # inner loop convergence tolerance
                 ):

        self.subproblems = subproblems
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.time = None
        self.con = con
        self.y = np.zeros_like(con(self.x)) # Lagrange multipliers
        self.history = [self.x.copy()] # data dictionary/list
        self.x_time = [0.0] # time history for each x update
        self.mu = np.ones(len(self.y)) if mu is None else mu # augmented Lagrangian penalty parameter
        self.max_mu = max_mu
        self.mu_history = [self.mu]

        assert rho >= 1
        self.rho = rho
        self.tau = tau
        self.tol = tol
        self.eps = eps


    def _update_mu(self, c_new, c_old):

        num = 0
        for i, (f_new, f_old) in enumerate(zip(abs(c_new), abs(c_old))):
            # if f_new > self.tau * f_old and f_new > self.tol:
            if f_new > self.tol:
                self.mu[i] = min(self.rho * self.mu[i], self.max_mu)
                num += 1

        return num

    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=10, # maximum number of inner iterations
              ) -> None:
        
        t1 = time.perf_counter()

        c_old = self.con(self.x)

        for k in range(max_outer_iter):

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 

                    self.x = subP(self.x, self.y, self.mu)
                    self.history.append(self.x.copy())
                    self.mu_history.append(self.mu)
                    self.x_time.append(time.perf_counter() - t1)

                # z_new = np.concatenate([xi.ravel() for xi in self.x])
                # step = z_new - z_old
                # relative_step = np.linalg.norm(step * self.scale)

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = abs(z_new - z_old)
                denominator = np.maximum(abs(z_old), abs(z_new))
                denominator = np.maximum(denominator, 1e-5)  # floor
                relative_step = step / denominator

                print(f"pr_itr={j:03d} | "f"rel_stp={max(relative_step):.3e} | ")

                if max(relative_step) <= self.eps:
                    print('-Primal loop converged!-')
                    break

            # Evaluate the constraints
            c_new = self.con(self.x)
            print(np.abs(c_new))
            feas = np.max(np.abs(c_new))

            if feas <= self.tol:
                print('-Dual loop converged with feasibility: ', feas, ' in ', k, ' dual iterations!-')
                break

            # self.y += self.mu * c_new # always update the multipliers
            self.y += np.diag(self.mu) @ c_new # always update the multipliers

            # # update mu on a per-scalar-constraint basis
            # nc_up = 0
            # for i in range(len(c_new)):
            #     feas_i = np.abs(c_new[i])
            #     feas_prev_i = np.abs(c_old[i])

            #     if feas_i > self.tau * feas_prev_i and feas_i > self.tol:
            #         self.mu[i] = min(self.rho * self.mu[i], self.max_mu)
            #         nc_up += 1

            nc_up = self._update_mu(c_new, c_old)
            c_old = c_new # oops, i forgot to update c_old...

            print(f"du_itr={k:03d} | "
                  f"feas={feas:.3e} | "
                  f"max mu={np.max(self.mu):.3e} | "
                  f"min mu={np.min(self.mu):.3e} | "
                  f"y={np.linalg.norm(self.y):.3e} | "
                  f"updated mu for {nc_up} constraints"
                  )

        self.time = time.perf_counter() - t1

        return None