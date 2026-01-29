import numpy as np
import matplotlib.pyplot as plt

def zakharov(x):
    x = np.array(x)
    d = len(x)
    
    s = np.sum(0.5 * np.arange(1, d+1) * x) / d
    
    y = np.sum(x**2) + s**2 + s**4
    return y



# Create a grid of points
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)

# Evaluate Zakharov function on the grid
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = zakharov([X1[i, j], X2[i, j]])


cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
