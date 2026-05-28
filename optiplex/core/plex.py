from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                #  constraint: Callable = None,
                 con: Callable = lambda v: np.zeros(0),
                 ):

        self.subproblems = subproblems
        # self.x = x_init
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.time = None
        self.con = con
        self.y = np.zeros_like(con(self.x)) # Lagrange multipliers
        self.d = len(self.y) # number of constraints
        self.history = [self.x.copy()] # data dictionary/list
        self.mu_history = []
        self.feas_history = []
        self.feas_time = []
        self.x_time = [0.0] # time history for each x update
        # self.m_time = [0.0] # time history for each multiplier update
    

    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=1000, # maximum number of inner iterations
              rho: float=1.2, # penalty increase factor
              eps_inner: float=1e-1, # inner loop convergence tolerance
              eps_outer: float=1e-4, # outer loop convergence tolerance
              tol: float=1e-6, # feasibility tolerance
              mu = 1.0, # augmented Lagrangian penalty coefficient
              ) -> None:
        
        assert rho > 1
        t1 = time.perf_counter()

        self.mu_history.append(mu)

        for k in range(max_outer_iter):

            x_old = np.concatenate([xi.ravel() for xi in self.x])

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 

                    self.x = subP(self.x, self.y, mu)
                    self.history.append(self.x.copy())
                    self.mu_history.append(mu)
                    self.x_time.append(time.perf_counter() - t1)

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
            c = self.con(self.x)
            feas = np.linalg.norm(c)
            self.feas_history.append(feas)
            self.feas_time.append(time.perf_counter() - t1)

            x_new = np.concatenate([xi.ravel() for xi in self.x])
            # r_norm_outer = np.linalg.norm(x_new - x_old)
            step = abs(x_new - x_old)
            denom = np.maximum(abs(x_old), abs(x_new))
            denom = np.maximum(denom, 1e-5) # floor
            rel_step = max(step / denom)

            if feas > tol:
                self.y = self.y + mu * c # Update the multipliers
                mu = rho * mu # Update the penalty coefficient

            print(f"du_itr={k:03d} | "
                  f"r_o={rel_step:.3e} | "
                  f"feas={feas:.3e} | "
                  f"mu={mu:.2f} | "
                  f"y={np.linalg.norm(self.y):.3e}"
                  )
            
            # Check outer loop convergence
            if rel_step <= eps_outer and feas <= tol:
                print('-Dual loop converged!-')
                break

        self.time = time.perf_counter() - t1

        return None