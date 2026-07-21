import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import modopt as mo
from beam_jax import Beam, CSTube
from lifting_line_jax_2 import LiftingLine
from optiplex import Plex2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

# aero setup
N = 31
b = 10.0
c_root = 1.0
c_tip = 0.65
rho_atm = 1.225
v_inf = 60
lifting_line = LiftingLine(N, b, c_root, c_tip, v_inf, rho_atm)

def aero_model(twist):

    coef = lifting_line.solve_lifting_line_model(twist)
    CD = lifting_line.compute_drag_coefficient(coef)
    aero_loads = lifting_line.compute_forces(coef)
    aero_loads = jnp.linalg.norm(aero_loads, axis=1)
    CL = lifting_line.compute_lift_coefficient(coef)
    lift = 0.5 * rho_atm * v_inf**2 * CL * lifting_line.S
    
    return CD, aero_loads, lift

# structures setup
num_nodes   = 31
mesh        = np.zeros((num_nodes, 3))
mesh[:, 1]  = lifting_line.y
fixed_nodes = [num_nodes // 2] # the center node is fixed in all DOFs
r = 0.2 * (lifting_line.chord[:-1] + lifting_line.chord[1:]) / 4 # radius of the tube as a function of chord
m0 = 1e3
load_factor = 3
safety_factor = 1.5
tip_disp_target = 0.1

def structures_model(aero_loads, thickness):

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 2].set(aero_loads * load_factor * safety_factor)

    cs = CSTube(radius=r, thickness=thickness)
    beam = Beam(mesh=mesh, E=69e9, G=26e9, rho=3000, cs=cs, F=F, fixed_nodes=fixed_nodes)
    u = beam.solve()
    u = jnp.linalg.norm(u[:, :3], axis=1)
    right_tip_disp, left_tip_disp = u[-1], u[0]

    mass = beam.mass + m0
    weight = mass * 9.81

    return right_tip_disp, left_tip_disp, weight

# initial design variable values
thickness0 = np.ones(num_nodes - 1) * 0.002
twist0 = np.ones(N) * np.deg2rad(5)

# run the aero model once to populate data dict
CD_init, aero_loads_init, _ = aero_model(twist0)

# run the structures model once to populate data dict
_, _, weight_init = structures_model(aero_loads_init, thickness0)

# central data repository for intermediate variables
data = {'aero_loads': aero_loads_init,
        'CD': CD_init,
        'weight': weight_init,
        }

x_init = [twist0, thickness0, aero_loads_init, weight_init]

cd_history = []

f_scale = 1e-4
w_scale = 1e-4

def con(x):

    twist = x[0]
    thickness = x[1]
    aero_loads_copy = x[2]
    weight_copy = x[3]

    aero_loads = data['aero_loads']
    weight = data['weight']

    return jnp.concatenate([f_scale * (aero_loads - aero_loads_copy), w_scale * jnp.array([weight - weight_copy])])


def aero_subproblem(x, y, mu):

    # print('Solving aerodynamic subproblem...', end=' ', flush=True)

    twist = x[0]
    thickness = x[1]
    aero_loads_copy = x[2]
    weight_copy = x[3]

    v0 = np.concatenate([twist, np.array([weight_copy])])

    weight = data['weight']

    def jax_obj(v):

        twist = v[:N]
        weight_copy = v[-1]

        CD, aero_loads, _ = aero_model(twist)
        
        # consensus constraints
        c = jnp.concatenate([f_scale * (aero_loads - aero_loads_copy), w_scale * jnp.array([weight - weight_copy])])

        # return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
        return 1e3 * CD + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c
    
    def jax_con(v):

        twist = v[:N]
        weight_copy = v[-1]

        _, _, lift = aero_model(twist)

        return jnp.array([lift - weight_copy])
    
    x_scaler = np.concatenate([np.ones(N) * 10, # twist scaler
                               np.array([1e-3]) # weight scaler
                               ])

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, x_scaler=x_scaler, cl=0, cu=0, c_scaler=1e-3)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x'] / x_scaler
    twist_solution = ans[:N]
    weight_copy_solution = ans[-1]

    # update the data dict
    CD, aero_loads, _ = aero_model(twist_solution)
    data['CD'] = CD
    data['aero_loads'] = aero_loads
    cd_history.append(CD)
    # print('CD: ', CD, ' obj: ', optimizer.results['fun'])

    return [twist_solution, thickness, aero_loads_copy, weight_copy_solution]

def struct_subproblem(x, y, mu):

    # print('Solving structural subproblem....', end=' ', flush=True)

    twist = x[0]
    thickness = x[1]
    aero_loads_copy = x[2]
    weight_copy = x[3]

    v0 = np.concatenate([thickness, aero_loads_copy])
    
    CD = data['CD']
    aero_loads = data['aero_loads']

    def jax_obj(v):

        thickness = v[:num_nodes - 1]
        aero_loads_copy = v[num_nodes - 1:]

        _, _, weight = structures_model(aero_loads_copy, thickness)

        # consensus constraints
        c = jnp.concatenate([f_scale * (aero_loads - aero_loads_copy), w_scale * jnp.array([weight - weight_copy])])

        # return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
        return 1e3 * CD + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c
    
    def jax_con(v):

        thickness = v[:num_nodes - 1]
        aero_loads_copy = v[num_nodes - 1:]

        u_r, u_l, _ = structures_model(aero_loads_copy, thickness)

        return jnp.array([u_r - tip_disp_target, 
                          u_l - tip_disp_target])
    
    tl = np.ones(num_nodes - 1) * 0.001     # min gauge
    tu = np.ones(num_nodes - 1) * r # np.inf    # thickness upper
    fl = np.ones(num_nodes) * -np.inf       # f lower
    fu = np.ones(num_nodes) * np.inf        # f upper
    xl = np.concatenate([tl, fl])
    xu = np.concatenate([tu, fu])
    t_scaler = np.ones(num_nodes - 1) * 1e2    # thickness scaler
    l_scaler = np.ones(num_nodes) * 1e-2       # f scaler
    x_scaler = np.concatenate([t_scaler, l_scaler])

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, xl=xl, xu=xu, x_scaler=x_scaler, cl=0, cu=0, c_scaler=1e1)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x'] / x_scaler
    thickness_solution = ans[:num_nodes - 1]
    aero_loads_copy_solution = ans[num_nodes - 1:]

    # update the data dict
    _, _, weight = structures_model(aero_loads_copy_solution, thickness_solution)
    data['weight'] = weight

    cd_history.append(CD)
    # print('CD: ', CD, ' obj: ', optimizer.results['fun'])

    return [twist, thickness_solution, aero_loads_copy_solution, weight_copy]



opt = Plex2(subproblems=[aero_subproblem, struct_subproblem],
            x_init=x_init,
            con=con,
            mu=np.ones(N + 1) * 10,
            max_mu=1e6,
            rho=1.2,
            tau=0.5,
            tol=0.3e-3, # outer loop feasibility
            eps=1e-1, # initial inner loop convergence
            eta=1e-3, # final inner loop convergence
            )

opt.solve(max_outer_iter=100, 
          max_inner_iter=50,
          )



# print('Lagrange multipliers: ', opt.y)
# print('Penalty parameters: ', opt.mu)
print('Total time (s): ', opt.tf)

solution = opt.x

twist = solution[0]
thickness = solution[1]
aero_loads_copy = solution[2]
weight_copy = solution[3]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lifting_line.y, twist)
ax2.plot(thickness)
plt.tight_layout()
plt.show()


aero_loads = data['aero_loads']
f_con = (aero_loads_copy - aero_loads) * f_scale
weight = data['weight']
w_con = (weight_copy - weight) * w_scale

print('f_con max abs: ', jnp.max(jnp.abs(f_con)))
print('w_con: ', w_con)


# solution = np.load('examples/aero_structural/new_solution.npz')
solution = np.load('examples/aero_structural/new_solution copy.npz')
x_star = np.concatenate([solution['twist'], solution['thickness']])

history_vecs = [np.concatenate(h[:2]) for h in opt.history]
error = [np.linalg.norm((x - x_star) / x_star) for x in history_vecs]

print('CD: ', cd_history[-1])

print('Error: ', error[-1])

# plt.semilogy(error)
# plt.xlabel('Iteration')
# plt.ylabel('Relative error')
# plt.show()

plt.semilogy(opt.x_time, error)
plt.xlabel('Time (s)')
plt.ylabel('Relative error')
plt.show()

# mu_hist = np.asarray(opt.mu_history)
# for i in range(mu_hist.shape[1]):
#     plt.semilogy(opt.x_time, mu_hist[:, i], label=f'mu[{i}]')
# plt.xlabel('Time (s)')
# plt.ylabel('Penalty parameter')
# plt.show()

plt.semilogy(opt.feasibility)
plt.xlabel('Iteration')
plt.ylabel('Feasibility')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lifting_line.y, solution['twist'], label='Reference twist')
ax1.plot(lifting_line.y, twist, label='Plex twist', marker='o')
ax1.set_xlabel('Spanwise location')
ax1.set_ylabel('Twist (rad)')
ax1.legend()
ax2.plot(solution['thickness'], label='Reference thickness')
ax2.plot(thickness, label='Plex thickness', marker='o')
ax2.set_xlabel('Spanwise location')
ax2.set_ylabel('Thickness (m)')
ax2.legend()
plt.tight_layout()
plt.show()

aero_loads = data['aero_loads']
plt.plot(aero_loads_copy, label='aero_loads_copy', marker='o')
plt.plot(aero_loads, label='aero_loads')
plt.legend()
plt.xlabel('Spanwise location')
plt.ylabel('Load (N)')
plt.show()

plt.plot(cd_history)
plt.xlabel('Iteration')
plt.ylabel('CD')
plt.show()

# np.savez('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.npz', 
#          error=error, 
#          mu_history=opt.mu_history, 
#          x_time=opt.x_time, 
#          feasibility=opt.feasibility,
#          multipliers=opt.y_history,
#          )