from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                 con: Callable = lambda v: np.zeros(0),
                 tol: float = 1e-3, # outer loop feasibility tolerance
                 mu: float = None, # positive penalty parameter(s)
                 max_mu: float = 1e3, # maximum penalty parameter
                 rho: float = 1.2, # penalty increase factor
                 ):

        self.subproblems = subproblems
        # self.x = x_init
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.time = None
        self.con = con
        self.initial_constraint_values = self.con(self.x)
        self.y = np.zeros_like(self.initial_constraint_values) # Lagrange multipliers
        self.d = len(self.y) # number of constraints
        self.history = [self.x.copy()] # data dictionary/list
        self.mu = np.ones_like(self.initial_constraint_values) if mu is None else mu # penalty parameter
        self.mu_history = [self.mu]
        self.feas_history = []
        self.feas_time = []
        self.x_time = [0.0] # time history for each x update
        self.tol = tol
        self.rho = rho
        self.max_mu = max_mu
    

    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=1000, # maximum number of inner iterations
              eps_inner: float=1e-1, # inner loop convergence tolerance
              eps_outer: float=1e-4, # outer loop convergence tolerance
              ) -> None:
        
        t0 = time.perf_counter()

        for k in range(max_outer_iter):

            x_old = np.concatenate([xi.ravel() for xi in self.x])

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 

                    self.x = subP(self.x, self.y, self.mu)
                    self.history.append(self.x.copy())
                    self.mu_history.append(self.mu)
                    self.x_time.append(time.perf_counter() - t0)

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = abs(z_new - z_old)
                denom = np.maximum(abs(z_old), abs(z_new))
                denom = np.maximum(denom, 1e-5) # floor
                rel_step = max(step / denom)

                print(f"pr_itr={j:03d} | "f"rel_stp={rel_step:.3e} | ")

                if rel_step <= eps_inner:
                    print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break


            # Exit for unconstrained problems
            if self.d == 0: break

            # Evaluate the constraints
            # c = self.con(self.x)
            # feas = np.linalg.norm(c)
            c_new = self.con(self.x)
            feas = np.max(abs(c_new))
            self.feas_history.append(feas)
            self.feas_time.append(time.perf_counter() - t0)

            x_new = np.concatenate([xi.ravel() for xi in self.x])
            # r_norm_outer = np.linalg.norm(x_new - x_old)
            step = abs(x_new - x_old)
            denom = np.maximum(abs(x_old), abs(x_new))
            denom = np.maximum(denom, 1e-5) # floor
            rel_step = max(step / denom)

            if feas > self.tol:
                self.y = self.y + self.mu * c_new # Update the multipliers
                self.mu = self.rho * self.mu # Update the penalty coefficient

            print(f"du_itr={k:03d} | "
                  f"r={rel_step:.3e} | "
                  f"feas={feas:.3e} | "
                  f"mu={self.mu:.3e} | "
                  f"y={np.linalg.norm(self.y):.3e} | "
                  )
            
            # Check outer loop convergence
            if rel_step <= eps_outer and feas <= self.tol:
                print('-Dual loop converged!-')
                break

        self.time = time.perf_counter() - t0

        return None