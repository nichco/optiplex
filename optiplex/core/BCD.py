from typing import List, Callable
import numpy as np
import time


class BCD():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                 solution: np.ndarray, # known solution for convergence criterion
                 eps: float = 1e-5, # inner loop convergence tolerance
                 ):

        self.subproblems = subproblems
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.solution = solution
        self.history = [self.x.copy()]
        self.x_time = [0.0]
        self.t0 = None
        self.tf = None
        self.eps = eps


    def solve(self, 
              max_iter: int=10,  # maximum number of iterations
              ) -> None:
        
        self.t0 = time.perf_counter()

        for j in range(max_iter):

                # z_old = np.concatenate([xi.ravel() for xi in self.x])

                # block coordinate descent loop
                for subP in self.subproblems: 

                    self.x = subP(self.x)
                    self.history.append(self.x.copy())
                    self.x_time.append(time.perf_counter() - self.t0)

                # z_new = np.concatenate([xi.ravel() for xi in self.x])
                # step = abs(z_new - z_old)
                # denom = np.maximum(abs(z_old), abs(z_new))
                # denom = np.maximum(denom, 1e-5) # floor
                # rel_step = max(step / denom)

                # print(f"pr_itr={j:03d} | "f"rel_stp={rel_step:.3e} | ")

                # if rel_step <= self.eps:
                #     print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                #     break
                z = np.concatenate([xi.ravel() for xi in self.x])
                relative_error = np.linalg.norm(z - self.solution) / np.linalg.norm(self.solution, ord=np.inf)
                print(f"pr_itr={j:03d} | "f"rel_err={relative_error:.3e} | ")

                if relative_error <= self.eps:
                    print('-Primal loop converged with relative error: ', relative_error, ' in ', j, ' iterations!-')
                    break

        self.tf = time.perf_counter() - self.t0

        return None