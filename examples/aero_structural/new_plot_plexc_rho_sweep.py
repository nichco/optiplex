import numpy as np
import matplotlib.pyplot as plt




def count_steps(vector, threshold=None, sigma=1.0):
    """
    Count the number of steps in a numpy vector.

    Args:
        vector:    1D numpy array with step-like behavior.
        threshold: Absolute difference to qualify as a step.
                   If None, auto-detects using mean + sigma * std of diffs.
        sigma:     Sensitivity multiplier for auto-threshold (lower = more sensitive).

    Returns:
        step_count:   Number of steps detected.
        step_indices: Indices where steps occur.
    """
    diffs = np.abs(np.diff(vector))

    if threshold is None:
        threshold = diffs.mean() + sigma * diffs.std()
        # print(f"Auto-threshold: {threshold:.4f}")

    step_indices = np.where(diffs > threshold)[0] + 1  # +1 to point to the start of the new level
    return len(step_indices), step_indices






r11 = np.load('examples/aero_structural/history4_r1p1.npz')
time_r11 = r11['x_time']
error_r11 = r11['error']
mu_r11 = r11['mu_history']
mu_r11 = np.asarray(mu_r11)
# print('Time for rho=1.1:', time_r11[-1], 's')
# print('Num iterations for rho=1.1:', len(time_r11))

all_step_indices = set()
for i in range(mu_r11.shape[1]):
    step_count, step_indices = count_steps(mu_r11[:, i], sigma=0.5)
    all_step_indices.update(step_indices)

total_steps = len(all_step_indices)
print(f"Dual iterations for rho=1.1: {total_steps}")

r13 = np.load('examples/aero_structural/history4_r1p3.npz')
time_r13 = r13['x_time']
error_r13 = r13['error']
mu_r13 = r13['mu_history']
mu_r13 = np.asarray(mu_r13)
# print('Time for rho=1.3:', time_r13[-1], 's')
# print('Num iterations for rho=1.3:', len(time_r13))

all_step_indices = set()
for i in range(mu_r13.shape[1]):
    step_count, step_indices = count_steps(mu_r13[:, i], sigma=0.5)
    all_step_indices.update(step_indices)

total_steps = len(all_step_indices)
print(f"Dual iterations for rho=1.3: {total_steps}")

r15 = np.load('examples/aero_structural/history4_r1p5.npz')
time_r15 = r15['x_time']
error_r15 = r15['error']
mu_r15 = r15['mu_history']
mu_r15 = np.asarray(mu_r15)
# print('Time for rho=1.5:', time_r15[-1], 's')
# print('Num iterations for rho=1.5:', len(time_r15))

all_step_indices = set()
for i in range(mu_r15.shape[1]):
    step_count, step_indices = count_steps(mu_r15[:, i], sigma=0.5)
    all_step_indices.update(step_indices)

total_steps = len(all_step_indices)
print(f"Dual iterations for rho=1.5: {total_steps}")

r20 = np.load('examples/aero_structural/history4_r2p0.npz')
time_r20 = r20['x_time']
error_r20 = r20['error']
mu_r20 = r20['mu_history']
mu_r20 = np.asarray(mu_r20)
# print('Time for rho=2.0:', time_r20[-1], 's')
# print('Num iterations for rho=2.0:', len(time_r20))

all_step_indices = set()
for i in range(mu_r20.shape[1]):
    step_count, step_indices = count_steps(mu_r20[:, i], sigma=0.5)
    all_step_indices.update(step_indices)

total_steps = len(all_step_indices)
print(f"Dual iterations for rho=2.0: {total_steps}")

r25 = np.load('examples/aero_structural/history4_r2p5.npz')
time_r25 = r25['x_time']
error_r25 = r25['error']
mu_r25 = r25['mu_history']
mu_r25 = np.asarray(mu_r25)
# print('Time for rho=2.5:', time_r25[-1], 's')
# print('Num iterations for rho=2.5:', len(time_r25))

all_step_indices = set()
for i in range(mu_r25.shape[1]):
    step_count, step_indices = count_steps(mu_r25[:, i], sigma=0.5)
    all_step_indices.update(step_indices)

total_steps = len(all_step_indices)
print(f"Dual iterations for rho=2.5: {total_steps}")

r30 = np.load('examples/aero_structural/history4_r3p0.npz')
time_r30 = r30['x_time']
error_r30 = r30['error']
mu_r30 = r30['mu_history']
mu_r30 = np.asarray(mu_r30)
# print('Time for rho=3.0:', time_r30[-1], 's')
# print('Num iterations for rho=3.0:', len(time_r30))

all_step_indices = set()
for i in range(mu_r30.shape[1]):
    step_count, step_indices = count_steps(mu_r30[:, i], sigma=0.5)
    all_step_indices.update(step_indices)

total_steps = len(all_step_indices)
print(f"Dual iterations for rho=3.0: {total_steps}")

plt.figure(figsize=(4, 3))
plt.semilogy(time_r11, error_r11, color='tab:blue', label=r'Error ($\rho=1.1$)', linewidth=2)
plt.semilogy(time_r13, error_r13, color='tab:orange', label=r'Error ($\rho=1.3$)', linewidth=2)
plt.semilogy(time_r15, error_r15, color='tab:green', label=r'Error ($\rho=1.5$)', linewidth=2)
plt.semilogy(time_r20, error_r20, color='tab:red', label=r'Error ($\rho=2.0$)', linewidth=2)
plt.semilogy(time_r25, error_r25, color='tab:purple', label=r'Error ($\rho=2.5$)', linewidth=2)
plt.semilogy(time_r30, error_r30, color='tab:brown', label=r'Error ($\rho=3.0$)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.legend()
plt.grid(axis='y', color='lavender')
# plt.savefig('simple_aero_struct_rho_sweep_2.pdf', bbox_inches='tight')
plt.show()