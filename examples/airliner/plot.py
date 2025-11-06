import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(h, r, v, gamma, m, eta, theta, tf):

    nu = len(eta)
    t0 = 0.0
    num_steps = len(h) - 1
    t = np.linspace(t0, tf, num_steps + 1)

    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax5 = plt.subplot2grid((3, 2), (2, 1))

    ax1.set_facecolor("ghostwhite")
    ax1.plot(r, h, linewidth=2.5, color='tab:blue')
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(color='lavender')
    ax1.set_yticks(np.arange(2e3, 15e3, 3e3))

    ax2.set_facecolor("ghostwhite")
    ax2.plot(np.linspace(t0, tf, nu), eta, color='tab:orange', linewidth=2)
    ax2.set_ylabel("Throttle")
    ax2.set_xlabel("Time (s)")
    ax2.grid(color='lavender')

    ax3.set_facecolor("ghostwhite")
    ax3.plot(np.linspace(t0, tf, nu), np.rad2deg(theta), color='tab:green', linewidth=2)
    ax3.set_ylabel("Pitch Angle (deg)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(color='lavender')

    ax4.set_facecolor("ghostwhite")
    ax4.plot(t, v, color='tab:red', linewidth=2, label='Velocity')
    ax4.set_ylabel("Velocity (m/s)")
    ax4.set_xlabel("Time (s)")
    ax4.grid(color='lavender')
    ax4.legend()

    ax5.set_facecolor("ghostwhite")
    ax5.plot(t, m, color='tab:brown', linewidth=2)
    ax5.set_ylabel("Mass (kg)")
    ax5.set_xlabel("Time (s)")
    ax5.grid(color='lavender')

    fig.suptitle('Trajectory', fontsize=12)
    fig.tight_layout()
    
    plt.show()




if __name__ == "__main__":

    import pickle

    with open('airliner/data.pkl', 'rb') as f:
        data = pickle.load(f)

    h = data['h']
    r = data['r']
    v = data['v']
    gamma = data['gamma']
    m = data['m']
    eta = data['eta']
    theta = data['theta']
    tf = data['tf']
    mach = data['mach']

    plot_trajectory(h, r, v, gamma, m, eta, theta, mach, tf)