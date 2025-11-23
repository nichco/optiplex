from typing import List, Callable
import numpy as np
import time
from sklearn.cluster import SpectralClustering
import os
os.environ["OMP_NUM_THREADS"] = '1' # avoid scikit-learn kmeans memory leak bug
import networkx as nx


class DuPlex():
    def __init__(self, 
                 blocks: List[Callable],
                 hessian: Callable,
                 x_init: List[np.ndarray],
                 constraint: Callable = None):
        
        # check for unconstrained problems
        self.con = False
        if constraint is not None:
            self.con = True

        self.blocks = blocks
        self.N = len(self.blocks)
        self.hessian = hessian
        self.x_init = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.solution = None
        self.num_iter = 0
        self.time = None
        self.constraint = constraint
        self.mu = 1.0 # augmented Lagrangian penalty coefficient

        if self.con:
            self.y = np.zeros_like(constraint(x_init)) # Lagrange multipliers
        else:
            self.y = None

        self.diffs = []
        self.constraint_violations = []

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

            print('PLEX ITR: ', k)

            x_k_minus_1 = self.x_init.copy()

            print('clustering...')
            H = self.hessian(self.x_init)

            # experimental avoid unconnected graphs??
            eps = 1e-6
            H = abs(H) + eps

            G = nx.Graph()

            # Add nodes
            for i in range(self.num_vars):
                G.add_node(i, label=f"x_{i+1}")

            for i in range(self.num_vars):
                for j in range(i + 1, self.num_vars):
                    G.add_edge(i, j, weight=H[i, j])


            adjacency_matrix = nx.adjacency_matrix(G).toarray()
    
            clustering = SpectralClustering(n_clusters=self.N,
                                            affinity="precomputed",
                                            # assign_labels="kmeans",
                                            assign_labels="kmeans",
                                            random_state=0,
                                            ).fit(adjacency_matrix)
            
            labels = clustering.labels_
            print('labels: ', labels)
            # exit()


            for block in self.blocks:
                self.x_init = block(self.x_init, self.y, self.mu, labels)


            if self.con:
                # Check convergence for constrained problems
                # Update the multipliers and penalty coefficient

                # evaluate the consensus constraint
                c = self.constraint(self.x_init)

                # Check convergence
                if all(np.allclose(new, old, rtol=tol) 
                    for new, old in zip(self.x_init, x_k_minus_1)) and all(np.abs(c) < ctol):
                    self.success = True
                    break

                # printing the convergence status
                max_diff = max([np.max(np.abs(new - old) / (np.abs(old) + 1e-12)) 
                        for new, old in zip(self.x_init, x_k_minus_1)])
                self.diffs.append(max_diff)
                max_constraint_violation = max(np.abs(c)) if len(c) > 0 else 0.0
                self.constraint_violations.append(max_constraint_violation)
                
                print('MAX DIFF: ', max_diff)
                print('LAGRANGE MULTIPLIERS: ', self.y)
                print('MAX CONSTRAINT VIOLATION: ', max_constraint_violation, 'CTOL: ', ctol)
                print('PENALTY COEFFICIENT: ', self.mu)

                # prevent overflow
                if any(np.abs(c) > ctol): # fixed a syntax error here

                    # Update the Lagrange multipliers
                    self.y = self.y + self.mu * c
                    
                    # Update the penalty coefficient
                    self.mu = rho * self.mu
                    print('NEW PENALTY COEFFICIENT: ', self.mu)



            else:

                max_diff = max([np.max(np.abs(new - old) / (np.abs(old) + 1e-12)) 
                        for new, old in zip(self.x_init, x_k_minus_1)])
                self.diffs.append(max_diff)
                
                print('MAX DIFF: ', max_diff)

                # Check convergence for unconstrained problems
                if all(np.allclose(new, old, rtol=tol) 
                    for new, old in zip(self.x_init, x_k_minus_1)):
                    self.success = True
                    break



        self.num_iter = k + 1
        self.time = time.time() - t1
        self.solution = self.x_init

        return self.success