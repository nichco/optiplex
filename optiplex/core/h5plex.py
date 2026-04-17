from typing import List, Callable
import numpy as np
import time
import h5py

class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                #  constraint: Callable = None,
                 con: Callable = lambda v: np.zeros(0),
                 path: str = "checkpoint.h5",
                 solution: np.ndarray = None,
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
        self.x_time = [0.0] # time history for each x update
        # self.m_time = [0.0] # time history for each multiplier update
        self.checkpoint_path = path
        self.solution = solution
    

    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=1000, # maximum number of inner iterations
              rho: float=1.2, # penalty increase factor
              ATOL_in: float=1e-1,
              RTOL_in: float=1e-1,
              ATOL_out: float=1e-4,
              RTOL_out: float=1e-4,
              ATOL_feas: float=1e-6,
              mu = 1.0, # augmented Lagrangian penalty coefficient
              ) -> None:
        
        assert rho > 1
        t1 = time.perf_counter()

        self.mu_history.append(mu)

        # with h5py.File(self.checkpoint_path, "w") as f:
        #     pass # clear the file from previous runs by opening in write mode and immediately closing

        itr = 0 # iteration counter for checkpointing in the h5py file
        # open the h5 file and extract the latest itration number to continue from if the file already exists and has data
        try:
            with h5py.File(self.checkpoint_path, 'r') as f:
                if len(f.keys()) > 0:
                    latest_iteration = sorted((int(key) for key in f.keys()))[-1]
                    itr = latest_iteration + 1
        except FileNotFoundError:
            pass # if the file doesn't exist, we'll create it during the first checkpoint save

        for k in range(max_outer_iter):

            x_old = np.concatenate([xi.ravel() for xi in self.x])

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 

                    self.x = subP(self.x, self.y, mu)
                    self.history.append(self.x.copy())
                    self.mu_history.append(mu)
                    self.x_time.append(time.perf_counter() - t1)


                    x_star = self.solution
                    denominator = np.where(np.abs(x_star) > 1e-12, x_star, 1.0) # prevent divide-by-zero errors
                    x_vec = np.concatenate([xi.ravel() for xi in self.x])
                    error = np.linalg.norm((x_vec - x_star) / denominator)

                    with h5py.File(self.checkpoint_path, "a") as f:  # "a" = append mode
                        iteration_group = f.create_group(str(itr))
                        iteration_group.create_dataset("x", data=x_vec)
                        iteration_group.create_dataset("error", data=error)
                        iteration_group.create_dataset("mu", data=mu)
                    itr += 1 # iteration counter for checkpointing in the h5py file

                eps_inner = np.sqrt(self.n) * ATOL_in + RTOL_in * np.linalg.norm(z_old)
                z_new = np.concatenate([xi.ravel() for xi in self.x])
                r_norm_inner = np.linalg.norm(z_new - z_old)

                # Print inner iteration data
                print(f"pr_itr={j:03d} | "
                      f"r_i={r_norm_inner:.3e} | "
                      )
                
                # Check inner loop convergence
                if r_norm_inner <= eps_inner: 
                    print('-Primal loop converged!-')
                    break
            
            # Exit for unconstrained problems
            if self.d == 0: break

            # Evaluate the constraints
            c = self.con(self.x)
            feas = np.linalg.norm(c)

            x_new = np.concatenate([xi.ravel() for xi in self.x])
            r_norm_outer = np.linalg.norm(x_new - x_old)

            # Outer loop convergence tolerance
            eps_outer = np.sqrt(self.n) * ATOL_out + RTOL_out * np.linalg.norm(x_old)
            eps_feas = np.sqrt(self.d) * ATOL_feas

            if feas > eps_feas:
                self.y = self.y + mu * c # Update the multipliers
                mu = rho * mu # Update the penalty coefficient
                # self.mu_history.append(mu)
                # self.m_time.append(time.perf_counter() - t1)

            print(f"du_itr={k:03d} | "
                  f"r_o={r_norm_outer:.3e} | "
                  f"feas={feas:.3e} | "
                  f"mu={mu:.2f} | "
                  f"y={np.linalg.norm(self.y):.3e}"
                  )
            
            # Check outer loop convergence
            if r_norm_outer <= eps_outer and feas <= eps_feas:
                print('-Dual loop converged!-')
                break


        self.time = time.perf_counter() - t1

        return None