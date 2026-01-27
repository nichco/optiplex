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
        self.x = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.k = 1
        self.time = None
        self.constraint = constraint
        self.mu = 1.0 # augmented Lagrangian penalty coefficient
        self.y = np.zeros_like(constraint(x_init)) # Lagrange multipliers
    

    def solve(self, 
              max_iter: int=100, # maximum number of outer iterations
              tol: float=1e-6, # outer loop tolerance
              rho: float=1.2, # penalty increase factor
              ctol: float=1e-4, # consensus constraint tolerance
              itol: float=1e2, # inner loop tolerance
              ) -> bool:
        
        assert rho > 1
        t1 = time.perf_counter()

        while self.success is False and self.k < max_iter:

            x_k_minus_1 = self.x.copy()

            inner_loop_converged, inner_loop_iterations = False, 0
            while not inner_loop_converged:
                print(f"  Inner Loop Iteration: {inner_loop_iterations+1}")

                x_j_minus_1 = self.x.copy()

                for block in self.blocks: 
                    self.x = block(self.x, self.y, self.mu)

                # Check inner loop convergence
                # if all(np.allclose(n, o, atol=itol, rtol=itol) for n, o in zip(self.x, x_j_minus_1)): 
                # if np.allclose(np.linalg.norm(np.concatenate(self.x)), np.linalg.norm(np.concatenate(x_j_minus_1)), atol=itol, rtol=itol): 
                if all(np.linalg.norm(n - o) <= itol * max(1.0, np.linalg.norm(o)) for n, o in zip(self.x, x_k_minus_1)):
                    inner_loop_converged = True

                inner_loop_iterations += 1


            # Evaluate the consensus constraints
            c = self.constraint(self.x)
            feasibility = np.linalg.norm(c)

            # Check outer loop convergence
            # if (all(np.allclose(n, o, rtol=tol) for n, o in zip(self.x, x_k_minus_1)) and feasibility < ctol): 
            #     self.success = True

            if all(np.linalg.norm(n - o) <= tol * max(1.0, np.linalg.norm(o)) for n, o in zip(self.x, x_k_minus_1)) and feasibility < ctol:
                self.success = True

            # If infeasible, update multipliers
            if feasibility > ctol:

                # Update the Lagrange multipliers
                self.y = self.y + self.mu * c
                    
                # Update the penalty coefficient
                self.mu = rho * self.mu


            # progress = np.max([np.max(np.abs(new - old) / (np.abs(old) + 1e-12)) 
            #                 for new, old in zip(self.x, x_k_minus_1)])
            
            progress = np.max([np.linalg.norm(n - o) for n, o in zip(self.x, x_k_minus_1)])


            df = pd.DataFrame({'iter': self.k,
                               'progress': progress,
                               '||c||': feasibility,
                               'mu': [self.mu],
                               '||y||': [np.linalg.norm(self.y)],
                               'cd iter': [inner_loop_iterations],
                               })

            # print(df.to_string(index=False, float_format='{:.3f}'.format))
            print(df.to_string(index=False,
                               formatters={
                               'progress': '{:.7f}'.format,   # more precision here
                               '||c||': '{:.5f}'.format,
                               'mu': '{:.3f}'.format,
                               '||y||': '{:.3f}'.format,
                               }
                               )
                 )

            # Update the iteration counter
            self.k += 1


        self.time = time.perf_counter() - t1

        return self.success