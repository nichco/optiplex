from typing import List, Callable
import numpy as np
import time
import pandas as pd

class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                #  constraint: Callable = None,
                 con: Callable = lambda x: np.zeros(0),
                 ):

        self.subproblems = subproblems
        # self.x = x_init
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.success = False
        self.dual_iterations = 1
        self.time = None
        self.con = con
        self.mu = 1.0 # augmented Lagrangian penalty coefficient
        self.y = np.zeros_like(con(self.x)) # Lagrange multipliers
    

    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=1000, # maximum number of inner iterations
              rho: float=1.2, # penalty increase factor
              ATOL_in: float=1e-1,
              RTOL_in: float=1e-1,
              ATOL_out: float=1e-4,
              RTOL_out: float=1e-4,
              EPS_pri: float=1e-6,
              ) -> bool:
        
        assert rho > 1
        t1 = time.perf_counter()


        for k in range(max_outer_iter):

            x_old = np.concatenate([xi.ravel() for xi in self.x])

            for j in range(max_inner_iter):
                print('inner iteration: ', j + 1)

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 
                    self.x = subP(self.x, self.y, self.mu)

                # Check inner loop convergence
                eps_inner = np.sqrt(self.n) * ATOL_in + RTOL_in * np.linalg.norm(z_old)
                z_new = np.concatenate([xi.ravel() for xi in self.x])
                if np.linalg.norm(z_new - z_old) < eps_inner: 
                    break

            # Evaluate the constraints
            c = self.con(self.x)
            feas = np.linalg.norm(c)

            x_new = np.concatenate([xi.ravel() for xi in self.x])
            r_norm = np.linalg.norm(x_new - x_old)

            # Outer loop convergence tolerance
            eps_outer = np.sqrt(self.n) * ATOL_out + RTOL_out * np.linalg.norm(x_old)
            
            # Check outer loop convergence
            if r_norm < eps_outer and feas < EPS_pri:
                self.success = True
                break

            if feas >= EPS_pri:
                self.y = self.y + self.mu * c # Update the multipliers
                self.mu = rho * self.mu # Update the penalty coefficient


            df = pd.DataFrame({'out_iter': self.dual_iterations,
                               '||r||': r_norm,
                               '||c||': feas,
                               'mu': [self.mu],
                               '||y||': [np.linalg.norm(self.y)],
                               'in_iter': j + 1,
                               })

            print(df.to_string(index=False, float_format='{:.3e}'.format))

        
        
        self.dual_iterations = k + 1

        self.time = time.perf_counter() - t1

        return self.success