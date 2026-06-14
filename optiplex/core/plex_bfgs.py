from typing import List, Callable
import numpy as np
import time


class Plex():
    def __init__(self, 
                 subproblems: List[Callable],
                 x_init: List[np.ndarray],
                 con: Callable = lambda v: np.zeros(0),
                 mu: np.ndarray | float = 1.0, # positive penalty parameter(s)
                 max_mu: float = 1e3, # maximum penalty parameter
                 rho: float = 1.2, # penalty increase factor
                 tau: float = 0.5, # factor for increasing mu based on constraint violation
                 tol: float = 1e-3, # outer loop feasibility tolerance
                 eps: float = 1e-5, # inner loop convergence tolerance
                 max_y: float = 1e6, # maximum Lagrange multiplier value
                 ):

        self.subproblems = subproblems
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        self.con = con

        self.initial_constraint_values = self.con(self.x)
        self.y = np.zeros_like(self.initial_constraint_values) # Lagrange multipliers

        m = self.y.size

        # inverse dual Hessian approximation
        if isinstance(mu, (float, int)):
            self.B = mu * np.eye(m)
        else:
            self.B = np.diag(mu)

        self.prev_y = None
        self.prev_g = None




        self.history = [self.x.copy()]
        self.feasibility = [max(abs(self.initial_constraint_values))]
        self.x_time = [0.0]
        self.y_history = [self.y.copy()]

        self.mu = mu # penalty parameter(s)
        assert np.all(self.mu >= 0)
        self.max_mu = max_mu
        # self.mu_history = [self.mu.copy()]
        self.mu_history = [self.mu]
        self.t0 = None
        self.tf = None

        assert rho >= 1
        self.rho = rho
        self.tau = tau
        self.tol = tol
        self.eps = eps
        self.max_y = max_y


    def _update_bfgs(self, y_new, g_new):

        if self.prev_y is None:
            self.prev_y = y_new.copy()
            self.prev_g = g_new.copy()
            return

        s = y_new - self.prev_y
        r = g_new - self.prev_g

        sty = float(s @ r)

        # curvature condition
        if sty > 1e-12:

            Bs = self.B @ r
            rBr = float(r @ Bs)

            if rBr > 1e-12:

                self.B += np.outer(s, s) / sty
                self.B -= np.outer(Bs, Bs) / rBr

        self.prev_y = y_new.copy()
        self.prev_g = g_new.copy()


    def _update_mu(self, c_new, c_old) -> None:

        if isinstance(self.mu, np.ndarray):

            num = 0
            for i, (f_new, f_old) in enumerate(zip(abs(c_new), abs(c_old))):
                if f_new > self.tau * f_old and f_new > self.tol:
                    self.mu[i] = min(self.rho * self.mu[i], self.max_mu)
                    num += 1

        elif isinstance(self.mu, (float, int)):

            if max(abs(c_new)) > self.tau * max(abs(c_old)) and max(abs(c_new)) > self.tol:
                self.mu = min(self.rho * self.mu, self.max_mu)

        return None


    def _inner_loop(self) -> None:
        
        for subP in self.subproblems: 

            self.x = subP(self.x, self.y, self.mu)
            self.history.append(self.x.copy())
            # self.mu_history.append(self.mu.copy())
            self.mu_history.append(self.mu)
            self.x_time.append(time.perf_counter() - self.t0)


    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=10,  # maximum number of inner iterations
              ) -> None:
        
        self.t0 = time.perf_counter()

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

                if rel_step <= self.eps:
                    print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break


            c_new = self.con(self.x)
            # print(abs(c_new))
            feas = np.max(abs(c_new))
            self.feasibility.append(feas)

            if feas <= self.tol:
                print('-Dual loop converged with feasibility: ', feas, ' in ', k, ' dual iterations!-')
                break

            # self.y += self.mu * c_new # always update the multipliers
            # self.y += np.diag(self.mu) @ c_new # always update the multipliers
            # if isinstance(self.mu, np.ndarray):
            #     self.y += np.diag(self.mu) @ c_new
            # if isinstance(self.mu, (float, int)):
            #     self.y += self.mu * c_new

            g = np.asarray(c_new, dtype=float)

            # quasi-Newton dual step
            dy = -self.B @ g

            y_old = self.y.copy()

            self.y += dy
            self.y = np.clip(self.y,
                            -self.max_y,
                            self.max_y)

            # update inverse Hessian approximation
            self._update_bfgs(self.y, g)

            self.y_history.append(self.y)


            self._update_mu(c_new, c_old)

            c_old = c_new

            print(f"du_itr={k:03d} | "
                  f"feas={feas:.3e} | "
                  f"max mu={np.max(self.mu):.3e} | "
                  f"min mu={np.min(self.mu):.3e} | "
                  f"y={np.linalg.norm(self.y):.3e} | "
                  )

        self.tf = time.perf_counter() - self.t0

        return None