import numpy as np
import matplotlib.pyplot as plt

N = 2
n = 30

solution = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N2.npz')
l_star = solution['l']
mp_star = solution['mp']
x_list_star = np.array(solution['x_list'])
u_list_star = np.array(solution['u_list'])

v_stars = []
for i in range(N):
    v_i_star = np.concatenate((np.array([l_star, mp_star]), x_list_star[i], u_list_star[i]))
    v_stars.append(v_i_star)

solution = np.concatenate(v_stars)

data = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N2.npz')
h = data['history']
h = np.array(h)
n_itr = len(h)

error = np.zeros(n_itr)
for i in range(n_itr):
    h_i = h[i].flatten()
    error_i = np.linalg.norm((h_i - solution))
    error[i] = error_i

plt.semilogy(error, linewidth=2)
plt.grid()
plt.show()

"""

l_history = np.zeros((n_itr, N))
mp_history = np.zeros((n_itr, N))
x_history = np.zeros((n_itr, N, 4 * n))
u_history = np.zeros((n_itr, N, n))

for itr in range(n_itr):
    v_itr = h[itr]
    
    for subp in range(N):
        v_subp = v_itr[subp]
        l_subp = v_subp[0]
        mp_subp = v_subp[1]
        x_subp = v_subp[2:2 + 4 * n]
        u_subp = v_subp[2 + 4 * n:2 + 5 * n]

        l_history[itr, subp] = l_subp
        mp_history[itr, subp] = mp_subp
        x_history[itr, subp, :] = x_subp
        u_history[itr, subp, :] = u_subp


plt.plot(l_history, label='l')
plt.plot(mp_history, label='mp')
plt.axhline(y=l_star, color='r', linestyle='--', label='l*')
plt.axhline(y=mp_star, color='g', linestyle='--', label='mp*')
plt.xlabel('Iteration')
plt.legend()
plt.show()

# l_error = np.linalg.norm((l_history - l_star) / l_star, axis=1)
# mp_error = np.linalg.norm((mp_history - mp_star) / mp_star, axis=1)
l_error = np.linalg.norm((l_history - l_star), axis=1)
mp_error = np.linalg.norm((mp_history - mp_star), axis=1)

x_error = np.zeros((n_itr, N))
u_error = np.zeros((n_itr, N))
for i in range(N):
    x_i = x_history[:, i, :]
    x_i_star = x_list_star[i]
    u_i = u_history[:, i, :]
    u_i_star = u_list_star[i]

    # x_error_i = np.linalg.norm((x_i - x_i_star) / x_i_star, axis=1)
    x_error_i = np.linalg.norm((x_i - x_i_star), axis=1)
    x_error[:, i] = x_error_i
    
    # u_error_i = np.linalg.norm((u_i - u_i_star) / u_i_star, axis=1)
    u_error_i = np.linalg.norm((u_i - u_i_star), axis=1)
    u_error[:, i] = u_error_i

plt.semilogy(l_error, label='l error')
plt.semilogy(mp_error, label='mp error')
plt.semilogy(x_error, label='x error')
plt.semilogy(u_error, label='u error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()



# plt.plot(x_list_star[0], label='x0*', color='black')
# plt.plot(x_list_star[1], label='x1*', color='black')
# plt.plot(x_history[-1, 0, :], label='x0')
# plt.plot(x_history[-1, 1, :], label='x1')
# plt.legend()
# plt.show()
"""