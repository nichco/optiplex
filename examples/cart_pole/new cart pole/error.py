import numpy as np
import matplotlib.pyplot as plt

N = 2
n = 30

solution = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N2.npz')
l_star = solution['l']
mp_star = solution['mp']
x_list_star = solution['x_list']
u_list_star = solution['u_list']

data = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N2.npz')
h = data['history']
n_itr = len(h)

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
# position_error = np.zeros((n_itr, N))
# theta_error = np.zeros((n_itr, N))
# dx_error = np.zeros((n_itr, N))
# dtheta_error = np.zeros((n_itr, N))
u_error = np.zeros((n_itr, N))
for i in range(N):
    x_i = x_history[:, i, :]
    x_i_star = x_list_star[i]
    u_i = u_history[:, i, :]
    u_i_star = u_list_star[i]

    # x_i = x_i.reshape((n_itr, 4, n))

    # position_i = x_i[:, 0, :]
    # theta_i = x_i[:, 1, :]
    # dx_i = x_i[:, 2, :]
    # dtheta_i = x_i[:, 3, :]

    # x_i_star = x_i_star.reshape((4, n))
    # position_i_star = x_i_star[0, :]
    # theta_i_star = x_i_star[1, :]
    # dx_i_star = x_i_star[2, :]
    # dtheta_i_star = x_i_star[3, :]

    # position_error_i = np.linalg.norm((position_i - position_i_star) / position_i_star, axis=1)
    # theta_error_i = np.linalg.norm((theta_i - theta_i_star) / theta_i_star, axis=1)
    # dx_error_i = np.linalg.norm((dx_i - dx_i_star) / dx_i_star, axis=1)
    # dtheta_error_i = np.linalg.norm((dtheta_i - dtheta_i_star) / dtheta_i_star, axis=1)

    # position_error[:, i] = position_error_i
    # theta_error[:, i] = theta_error_i
    # dx_error[:, i] = dx_error_i
    # dtheta_error[:, i] = dtheta_error_i

    # x_error_i = np.linalg.norm((x_i - x_i_star) / x_i_star, axis=1)
    x_error_i = np.linalg.norm((x_i - x_i_star), axis=1)
    x_error[:, i] = x_error_i
    
    # u_error_i = np.linalg.norm((u_i - u_i_star) / u_i_star, axis=1)
    u_error_i = np.linalg.norm((u_i - u_i_star), axis=1)
    u_error[:, i] = u_error_i

plt.semilogy(l_error, label='l error')
plt.semilogy(mp_error, label='mp error')
plt.semilogy(x_error, label='x error')
# plt.semilogy(position_error, label='position error')
# plt.semilogy(theta_error, label='theta error')
# plt.semilogy(dx_error, label='dx error')
# plt.semilogy(dtheta_error, label='dtheta error')
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