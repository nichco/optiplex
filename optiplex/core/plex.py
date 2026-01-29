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
              itol: float=1e2, # inner loop tolerance
              ) -> bool:
        
        assert rho > 1
        t1 = time.perf_counter()

        while self.success is False and self.dual_iterations < max_iter:

            x_out_minus_1 = np.concatenate([xi.ravel() for xi in self.x])

            inner_loop_converged, inner_loop_iterations = False, 0
            while not inner_loop_converged:
                print(f"  inner loop iteration: {inner_loop_iterations + 1}")

                x_in_minus_1 = np.concatenate([xi.ravel() for xi in self.x])

                for block in self.blocks: 
                    self.x = block(self.x, self.y, self.mu)

                # Check inner loop convergence
                x_in = np.concatenate([xi.ravel() for xi in self.x])
                # if np.linalg.norm(x_in - x_in_minus_1) < itol:
                if np.allclose(np.linalg.norm(x_in), np.linalg.norm(x_in_minus_1), atol=itol, rtol=itol):
                    inner_loop_converged = True

                inner_loop_iterations += 1


            # Evaluate the consensus constraints
            c = self.constraint(self.x)
            feasibility = np.linalg.norm(c)

            x_out = np.concatenate([xi.ravel() for xi in self.x])

            outer_loop_delta = np.linalg.norm(x_out - x_out_minus_1)

            rhs = tol + tol * np.linalg.norm(x_out_minus_1)
            # absolute(a - b) <= (atol + rtol * absolute(b))
            outer_loop_progress = outer_loop_delta - rhs
            
            # Check outer loop convergence
            if np.allclose(np.linalg.norm(x_out), np.linalg.norm(x_out_minus_1), atol=tol, rtol=tol) and feasibility < ctol:
                self.success = True

            if feasibility > ctol: # If infeasible, update multipliers
                self.y = self.y + self.mu * c # Update the multipliers
                self.mu = rho * self.mu # Update the penalty coefficient


            df = pd.DataFrame({'iter': self.dual_iterations,
                               'progress': outer_loop_progress,
                               '||c||': feasibility,
                               'mu': [self.mu],
                               '||y||': [np.linalg.norm(self.y)],
                               'cd iter': [inner_loop_iterations],
                               })

            print(df.to_string(index=False, float_format='{:.3e}'.format))

            self.dual_iterations += 1


        self.time = time.perf_counter() - t1

        return self.success