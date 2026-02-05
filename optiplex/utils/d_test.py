import numpy as np

def build_incidence_complete(N):
    i, j = np.triu_indices(N, k=1)
    print(i, j)
    M = i.shape[0]

    D = np.zeros((M, N))
    idx = np.arange(M)
    D[idx, i] = 1.0
    D[idx, j] = -1.0

    return D



D = build_incidence_complete(3)
print(D)