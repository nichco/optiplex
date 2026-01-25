from typing import List, Callable
import numpy as np
import time
from optiplex.utils.format import HEADER, ROW_FMT


class Plex2():
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
        self.solution = None
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

            done = False
            while not done:

                for block in self.blocks:
                    self.x = block(self.x, self.y, self.mu)

                # # Check inner loop convergence
                if all(np.allclose(n, o, rtol=itol) for n, o in zip(self.x, x_k_minus_1)): done = True



            c = self.constraint(self.x)
            if (all(np.allclose(n, o, rtol=tol) for n, o in zip(self.x, x_k_minus_1)) 
                and all(np.abs(c) < ctol)): self.success = True

            # prevent overflow
            if any(np.abs(c) > ctol):

                # Update the Lagrange multipliers
                self.y = self.y + self.mu * c
                    
                # Update the penalty coefficient
                self.mu = rho * self.mu


            progress = max([np.max(np.abs(new - old) / (np.abs(old) + 1e-12)) 
                            for new, old in zip(self.x, x_k_minus_1)])

            max_constraint_violation = max(np.abs(c)) if len(c) > 0 else 0.0

            if self.k % 3 == 0: print(HEADER)

            print(ROW_FMT.format(
                iter=self.k,
                progress=progress,
                max_constr=max_constraint_violation,
                ctol=ctol,
                mu=self.mu,
                y_norm=np.linalg.norm(self.y),
            ))

            # Update the iteration counter
            self.k += 1


        self.time = time.perf_counter() - t1
        self.solution = self.x

        return self.success