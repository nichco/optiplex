from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 blocks: List[Callable],
                 constraint: Callable,
                 x_init: List[np.ndarray]):
        
        self.blocks = blocks
        self.x_init = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.solution = None
        self.num_iter = 0
        self.time = None
        self.constraint = constraint
        self.mu = 1.0 # augmented Lagrangian penalty coefficient
        self.y = np.zeros_like(constraint(x_init)) # Lagrange multipliers

    def solve(self, 
              max_iter: int=100, 
              tol: float=1e-6,
              rho: float=1.2, # penalty increase factor
              ctol: float=1e-4, # consensus constraint tolerance
              ) -> bool:
        
        # check if rho is greater than 1
        if rho <= 1: raise ValueError("rho must be greater than 1")


        t1 = time.time()

        for k in range(max_iter):

            x_k_minus_1 = self.x_init.copy()

            for block in self.blocks:
                self.x_init = block(self.x_init, self.y, self.mu)



            # evaluate the consensus constraint
            c = self.constraint(self.x_init)

            # Check convergence
            if all(np.allclose(new, old, rtol=tol) 
                   for new, old in zip(self.x_init, x_k_minus_1)) and all(np.abs(c) < ctol):
                self.success = True
                break

            # prevent overflow
            if any(np.abs(c)) > ctol:

                # Update the Lagrange multipliers
                self.y = self.y + self.mu * c
                
                # Update the penalty coefficient
                self.mu = rho * self.mu


        self.num_iter = k + 1
        self.time = time.time() - t1
        self.solution = self.x_init

        return self.success