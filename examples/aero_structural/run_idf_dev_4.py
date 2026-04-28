import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import modopt as mo
from beam_jax import Beam, CSTube
from lifting_line_jax_2 import LiftingLine
from optiplex import PlexC
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
    CD = lifting_line.compute_drag(coef)
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
    beam = Beam(mesh=mesh, E=69e9, G=26e9, rho=3000,
                A=cs.area, J=cs.J, Iy=cs.Iy, Iz=cs.Iz, F=F, 
                fixed_nodes=fixed_nodes)
    u = beam.solve()
    u = jnp.linalg.norm(u[:, :3], axis=1)
    right_tip_disp, left_tip_disp = u[-1], u[0]

    mass = beam.mass + m0
    weight = mass * 9.81

    return right_tip_disp, left_tip_disp, weight


# scalers for constraint functions
lw_scale = 1e-4#2e-4 # increasing this further results in huge numbers of inner iterations
f_scale = 2e-4
disp_scale = 3e-1

# initial design variable values
thickness0 = np.ones(num_nodes - 1) * 0.002
twist0 = np.ones(N) * np.deg2rad(5)

# run the aero model once to populate data dict
CD_init, aero_loads_init, lift_init = aero_model(twist0)

# run the structures model once to populate data dict
right_tip_disp_init, left_tip_disp_init, weight_init = structures_model(aero_loads_init, thickness0)

# central data repository for intermediate variables
data = {'aero_loads': aero_loads_init,
        'CD': CD_init,
        'Lift': lift_init,
        'right_tip_disp': right_tip_disp_init,
        'left_tip_disp': left_tip_disp_init,
        'Weight': weight_init,
        }

# a list of initial variable values for Plex
x_init = [twist0, thickness0, aero_loads_init]

# record the objective value history for plotting
cd_history = []

def con(x):

    aero_loads_copy = x[2]
    aero_loads = data['aero_loads']
    lift = data['Lift']
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    f_con = (aero_loads_copy - aero_loads) * f_scale # consensus constraint
    l_equals_w = (lift - weight) * lw_scale # lift equals weight constraint
    right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale # displacement constraints
    left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale # displacement constraints

    return jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])


def aero_subproblem(x, y, mu):

    print('Solving aerodynamic subproblem...', end=' ', flush=True)

    twist = x[0]
    thickness = x[1]
    aero_loads_copy = x[2]
    v0 = twist
    
    # unpack the data dict
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    def jax_obj(v):

        # run the aero model
        CD, aero_loads, lift = aero_model(v)
        
        # compute the global constraints
        f_con = (aero_loads_copy - aero_loads) * f_scale
        l_equals_w = (lift - weight) * lw_scale
        right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
        left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale
        c = jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])

        # return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
        return 1e3 * CD + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c
    
    x_scaler = np.ones(N) * 10 # twist scaler

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    twist_solution = optimizer.results['x'] / x_scaler

    # update the data dict
    CD, aero_loads, lift = aero_model(twist_solution)
    data['aero_loads'] = aero_loads
    data['CD'] = CD
    data['Lift'] = lift
    cd_history.append(CD)
    print('CD: ', CD, ' obj: ', optimizer.results['fun'])

    return [twist_solution, thickness, aero_loads_copy]

def struct_subproblem(x, y, mu):

    print('Solving structural subproblem....', end=' ', flush=True)

    twist = x[0]
    t = x[1]
    aero_loads_copy = x[2]
    v0 = np.concatenate([t, aero_loads_copy])
    
    # unpack the data dict
    aero_loads = data['aero_loads']
    CD = data['CD']
    lift = data['Lift']

    def jax_obj(v):

        t = v[:num_nodes - 1]
        aero_loads_copy = v[num_nodes - 1:]

        # run the structures model
        right_tip_disp, left_tip_disp, weight = structures_model(aero_loads_copy, t)

        # compute the global constraints
        f_con = (aero_loads_copy - aero_loads) * f_scale
        l_equals_w = (lift - weight) * lw_scale
        right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
        left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale
        c = jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])

        # return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
        return 1e3 * CD + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c
    
    tl = np.ones(num_nodes - 1) * 0.001     # min gauge
    tu = np.ones(num_nodes - 1) * r # np.inf    # thickness upper
    fl = np.ones(num_nodes) * -np.inf       # f lower
    fu = np.ones(num_nodes) * np.inf        # f upper
    xl = np.concatenate([tl, fl])
    xu = np.concatenate([tu, fu])
    t_scaler = np.ones(num_nodes - 1) * 1e2    # thickness scaler
    l_scaler = np.ones(num_nodes) * 1e-2       # f scaler
    x_scaler = np.concatenate([t_scaler, l_scaler])

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    sol = optimizer.results['x'] / x_scaler
    thickness_solution = sol[:num_nodes - 1]
    load_solution = sol[num_nodes - 1:]

    # update the data dict
    right_tip_disp, left_tip_disp, weight = structures_model(load_solution, thickness_solution)
    data['right_tip_disp'] = right_tip_disp
    data['left_tip_disp'] = left_tip_disp
    data['Weight'] = weight

    cd_history.append(CD)
    print('CD: ', CD, ' obj: ', optimizer.results['fun'])

    return [twist, thickness_solution, load_solution]



opt = PlexC(subproblems=[aero_subproblem, struct_subproblem],
            x_init=x_init,
            con=con,
            mu=np.ones(N + 3) * 10, # initial penalty parameters for each constraint
            max_mu=1e6,
            rho=1.5,
            tau=0.5,#0.7
            tol=1e-3, # outer loop feasibility
            eps=1e-5, # inner loop convergence
            )

opt.solve(max_outer_iter=100, 
          max_inner_iter=100,
          )


# The optimal CD should be:  0.012349323882890414

print('Lagrange multipliers: ', opt.y)
print('Penalty parameters: ', opt.mu)

solution = opt.x

twist = solution[0]
thickness = solution[1]
aero_loads_copy = solution[2]


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lifting_line.y, twist)
ax2.plot(thickness)
plt.tight_layout()
plt.show()


lift = data['Lift']
weight = data['Weight']
right_tip_disp = data['right_tip_disp']
left_tip_disp = data['left_tip_disp']
print('Lift: ', lift)
print('Weight: ', weight)
print('Right tip displacement: ', right_tip_disp)
print('Left tip displacement: ', left_tip_disp)

aero_loads = data['aero_loads']
f_con = ((aero_loads_copy - aero_loads) / aero_loads) * f_scale
l_equals_w = (lift - weight) * lw_scale
right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale

print('Scaled constraint values: ')
print('f_con max abs: ', jnp.max(jnp.abs(f_con)))
print('l_equals_w: ', l_equals_w)
print('right_disp_con: ', right_disp_con)
print('left_disp_con: ', left_disp_con)


solution = np.load('examples/aero_structural/new_solution.npz')
x_star = np.concatenate([solution['twist'], solution['thickness']])

history_vecs = [np.concatenate(h[:2]) for h in opt.history]
error = [np.linalg.norm((x - x_star) / x_star) for x in history_vecs]

print('CD: ', cd_history[-1])

# plt.semilogy(error)
# plt.xlabel('Iteration')
# plt.ylabel('Relative error')
# plt.show()

plt.semilogy(opt.x_time, error)
plt.xlabel('Time (s)')
plt.ylabel('Relative error')
plt.show()

mu_hist = np.asarray(opt.mu_history)
for i in range(mu_hist.shape[1]):
    plt.semilogy(opt.x_time, mu_hist[:, i], label=f'mu[{i}]')
# plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Penalty parameter')
plt.show()

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

np.savez('examples/aero_structural/history4.npz', error=error, mu_history=opt.mu_history, x_time=opt.x_time)