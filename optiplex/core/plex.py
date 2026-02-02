from typing import List, Callable
import numpy as np
import time
import pandas as pd

class Plex():
    def __init__(self, 
                 blocks: List[Callable],
                 x_init: List[np.ndarray],
                #  constraint: Callable = None,
                 constraint: Callable = lambda x: 0,
                 ):

        self.blocks = blocks
        # self.x = x_init
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.success = False
        self.dual_iterations = 1
        self.time = None
        self.constraint = constraint
        self.mu = 1.0 # augmented Lagrangian penalty coefficient
        self.y = np.zeros_like(constraint(self.x)) # Lagrange multipliers
    

    def solve(self, 
              max_iter: int=100, # maximum number of outer iterations
              tol: float=1e-6, # outer loop tolerance
              rho: float=1.2, # penalty increase factor
              ctol: float=1e-4, # consensus constraint tolerance
              itol: float=1e3, # inner loop tolerance
              ABSTOL_in: float=1e-3,
              RELTOL_in: float=1e-2,
            #   ABSTOL_out: float=1e-6,
            #   RELTOL_out: float=1e-6,
              ) -> bool:
        
        assert rho > 1
        t1 = time.perf_counter()

        # while self.success is False and self.dual_iterations < max_iter:
        for k in range(max_iter):

            x_old = np.concatenate([xi.ravel() for xi in self.x])

            inner_loop_converged, primal_iterations = False, 0
            while not inner_loop_converged:
                print(f"  inner loop iteration: {primal_iterations + 1}")

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for block in self.blocks: 
                    self.x = block(self.x, self.y, self.mu)

                # Check inner loop convergence
                eps_primal = ABSTOL_in + RELTOL_in * np.linalg.norm(z_old)
                z_new = np.concatenate([xi.ravel() for xi in self.x])
                r_norm_inner = np.linalg.norm(z_new - z_old)

                if r_norm_inner < eps_primal:
                    inner_loop_converged = True

                primal_iterations += 1


            # Evaluate the consensus constraints
            c = self.constraint(self.x)
            feasibility = np.linalg.norm(c)

            x_new = np.concatenate([xi.ravel() for xi in self.x])

            r_norm_outer = np.linalg.norm(x_new - x_old)

            ABSTOL_out = RELTOL_out = tol
            eps_primal = ABSTOL_out + RELTOL_out * np.linalg.norm(x_old)
            
            # Check outer loop convergence
            if r_norm_outer < eps_primal and feasibility < ctol:
                self.success = True
                break

            if feasibility >= ctol: # If infeasible, update multipliers
                self.y = self.y + self.mu * c # Update the multipliers
                self.mu = rho * self.mu # Update the penalty coefficient


            df = pd.DataFrame({'iter': self.dual_iterations,
                               '||r||': r_norm_outer,
                               '||c||': feasibility,
                               'mu': [self.mu],
                               '||y||': [np.linalg.norm(self.y)],
                               'cd iter': [primal_iterations],
                               })

            print(df.to_string(index=False, float_format='{:.3e}'.format))

        
        
        self.dual_iterations = k + 1

        self.time = time.perf_counter() - t1

        return self.success