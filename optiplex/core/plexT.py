from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                #  constraint: Callable = None,
                 con: Callable = lambda v: np.zeros(0),
                 scale: float = 1.0,
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
        self.scale = scale

    def solve(self, 
              max_outer_iter: int=1000, # maximum number of outer iterations
              max_inner_iter: int=1000, # maximum number of inner iterations
              rho: float=1.2, # penalty increase factor
            #   ATOL_in: float=1e-1,
            #   RTOL_in: float=1e-1,
              eps: float=1e-1,
            #   ATOL_out: float=1e-4,
            #   RTOL_out: float=1e-4,
              tol: float=1e-6,
              mu = 1.0, # augmented Lagrangian penalty coefficient
              max_mu: float = 1000.0,
              tau: float = 0.5, # factor for increasing mu based on constraint violation
              ) -> None:
        
        # assert rho > 1
        t1 = time.perf_counter()

        self.mu_history.append(mu)

        feas_prev = np.linalg.norm(self.con(self.x))
        # feas_prev = np.inf

        for k in range(max_outer_iter):

            # x_old = np.concatenate([xi.ravel() for xi in self.x])

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                for subP in self.subproblems: 

                    self.x = subP(self.x, self.y, mu)
                    self.history.append(self.x.copy())
                    self.mu_history.append(mu)
                    self.x_time.append(time.perf_counter() - t1)

                # eps_inner = np.sqrt(self.n) * ATOL_in + RTOL_in * np.linalg.norm(z_old)
                # z_new = np.concatenate([xi.ravel() for xi in self.x])
                # r_norm_inner = np.linalg.norm(z_new - z_old)

                # # Print inner iteration data
                # print(f"pr_itr={j:03d} | "
                #       f"r_i={r_norm_inner:.3e} | "
                #       )
                
                # # Check inner loop convergence
                # if r_norm_inner <= eps_inner: 
                #     print('-Primal loop converged!-')
                #     break

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = z_new - z_old
                # scale = np.maximum(1.0, np.abs(z_old))
                # scale = np.maximum(self.scale, np.abs(z_old))
                # relative_step = np.linalg.norm(step / scale)
                relative_step = np.linalg.norm(step * self.scale)

                # Print inner iteration data
                print(f"pr_itr={j:03d} | "f"rel_stp={relative_step:.3e} | ")

                if relative_step <= eps:
                    print('-Primal loop converged!-')
                    break

            # Evaluate the constraints
            c = self.con(self.x)
            feas = np.linalg.norm(c)

            if feas <= tol:
                print('-Dual loop converged!-')
                break

            self.y += mu * c # Always update the multipliers
            # mu = rho * mu    # Update the penalty coefficient
            # mu = min(rho * mu, max_mu)



            if feas > tau * feas_prev:
                print('Increasing mu')
                mu = min(rho * mu, max_mu)

            feas_prev = feas



            print(f"du_itr={k:03d} | "
                  f"feas={feas:.3e} | "
                  f"mu={mu:.2f} | "
                  f"y={np.linalg.norm(self.y):.3e}"
                  )

            """
            x_new = np.concatenate([xi.ravel() for xi in self.x])
            r_norm_outer = np.linalg.norm(x_new - x_old)

            # Outer loop convergence tolerance
            eps_outer = np.sqrt(self.n) * ATOL_out + RTOL_out * np.linalg.norm(x_old)
            eps_feas = np.sqrt(self.d) * ATOL_feas

            if feas > eps_feas:
                self.y += mu * c # Update the multipliers
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
            """

        print('feasibility: ', feas)


        self.time = time.perf_counter() - t1

        return None