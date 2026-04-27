from typing import List, Callable
import numpy as np
import time
import h5py

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
                 eps: float = 1e-5, # inner loop convergence tolerance
                 path: str = "checkpoint.h5", # path for h5py
                 solution: np.ndarray = None, # solution for error calculation
                 ):

        self.subproblems = subproblems
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.con = con

        self.initial_constraint_values = self.con(self.x)
        self.y = np.zeros_like(self.initial_constraint_values) # Lagrange multipliers
        self.history = [self.x.copy()]
        self.feasibility = [max(abs(self.initial_constraint_values))]
        self.x_time = [0.0]
        self.mu = np.ones_like(self.initial_constraint_values) if mu is None else mu # penalty parameter
        assert np.all(self.mu >= 0)
        self.max_mu = max_mu
        self.mu_history = [self.mu.copy()]
        self.t0 = None
        self.tf = None

        assert rho >= 1
        self.rho = rho
        self.tau = tau
        self.tol = tol
        self.eps = eps

        self.checkpoint_path = path
        self.solution = solution
        self.itr = 0 # iteration counter for the h5py file


    def _update_mu(self, c_new, c_old) -> int:

        num = 0
        for i, (f_new, f_old) in enumerate(zip(abs(c_new), abs(c_old))):
            if f_new > self.tau * f_old and f_new > self.tol:
            # if f_new > self.tol:
                self.mu[i] = min(self.rho * self.mu[i], self.max_mu)
                num += 1

        return num
    

    def _inner_loop(self) -> None:
        
        for subP in self.subproblems: 

            self.x = subP(self.x, self.y, self.mu)
            self.history.append(self.x.copy())
            self.mu_history.append(self.mu.copy())
            self.x_time.append(time.perf_counter() - self.t0)

            x_star = self.solution
            denominator = np.where(np.abs(x_star) > 1e-12, x_star, 1.0) # prevent divide-by-zero errors
            x_vec = np.concatenate([xi.ravel() for xi in self.x])
            error = np.linalg.norm((x_vec - x_star) / denominator)

            with h5py.File(self.checkpoint_path, "a") as f:  # "a" = append mode
                        iteration_group = f.create_group(str(self.itr))
                        iteration_group.create_dataset("x", data=x_vec)
                        iteration_group.create_dataset("error", data=error)
                        iteration_group.create_dataset("mu", data=self.mu)
                        iteration_group.create_dataset("y", data=self.y)
                        iteration_group.create_dataset("time", data=time.perf_counter() - self.t0)

            self.itr += 1 # iteration counter for checkpointing in the h5py file

    
        def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=10,  # maximum number of inner iterations
              ) -> None:
            
            self.t0 = time.perf_counter()

            with h5py.File(self.checkpoint_path, "w") as f:
                pass # clear the file from previous runs by opening in write mode and immediately closing

            c_old = self.initial_constraint_values

            for k in range(max_outer_iter):

                for j in range(max_inner_iter):

                    z_old = np.concatenate([xi.ravel() for xi in self.x])

                    self._inner_loop()

                    z_new = np.concatenate([xi.ravel() for xi in self.x])
                    step = abs(z_new - z_old)
                    denom = np.maximum(abs(z_old), abs(z_new))
                    denom = np.maximum(denom, 1e-5) # floor
                    rel_step = max(step / denom)

                    print(f"pr_itr={j:03d} | "f"rel_stp={rel_step:.3e} | ")

                    if rel_step <= self.eps:
                        print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                        break


                c_new = self.con(self.x)
                print(abs(c_new))
                feas = np.max(abs(c_new))
                self.feasibility.append(feas)

                if feas <= self.tol:
                    print('-Dual loop converged with feasibility: ', feas, ' in ', k, ' dual iterations!-')
                    break

                self.y += np.diag(self.mu) @ c_new # always update the multipliers

                nc_up = self._update_mu(c_new, c_old)
                c_old = c_new

                print(f"du_itr={k:03d} | "
                    f"feas={feas:.3e} | "
                    f"max mu={np.max(self.mu):.3e} | "
                    f"min mu={np.min(self.mu):.3e} | "
                    f"y={np.linalg.norm(self.y):.3e} | "
                    f"updated mu for {nc_up} constraints"
                    )

            self.tf = time.perf_counter() - self.t0

            return None