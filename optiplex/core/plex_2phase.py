from typing import List, Callable
import numpy as np
import time


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
                 eps: float = 1e-5, # initial inner loop convergence tolerance
                 eta: float = 1e-3, # final inner loop convergence tolerance
                 max_y: float = 1e6, # maximum Lagrange multiplier value
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
        self.y_history = [self.y.copy()]
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
        self.eta = eta
        self.max_y = max_y


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


    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=10,  # maximum number of inner iterations
              ) -> None:
        
        phase = 0 # two-phase inner loop tolerance (ALGENCAN)
        
        self.t0 = time.perf_counter()
        # c_old = self.initial_constraint_values WRONG PLACE????

        for k in range(max_outer_iter):

            c_old = self.con(self.x)

            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                # block coordinate descent inner loop
                self._inner_loop()

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = abs(z_new - z_old)
                denom = np.maximum(abs(z_old), abs(z_new))
                denom = np.maximum(denom, 1e-5) # floor
                rel_step = max(step / denom)

                print(f"pr_itr={j:03d} | "f"rel_stp={rel_step:.3e} | ")

                if phase == 0 and rel_step <= self.eps:
                    print('-(Phase 0) primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break

                if phase == 1 and rel_step <= self.eta:
                    print('-(Phase 1) primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break

                # if rel_step <= self.eps:
                #     print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                #     break


            c_new = self.con(self.x)
            feas = np.max(abs(c_new))
            self.feasibility.append(feas)

            if feas <= self.tol and phase == 1:
                print('-Dual loop converged with feasibility: ', feas, ' in ', k, ' dual iterations!-')
                break

            if feas <= self.tol and phase == 0:
                print('-Phase 1 complete. Starting phase 2 with feasibility: ', feas)
                phase = 1

            self.y += np.diag(self.mu) @ c_new # always update the multipliers
            self.y_history.append(self.y.copy())

            # update mu on a per-scalar-constraint basis
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