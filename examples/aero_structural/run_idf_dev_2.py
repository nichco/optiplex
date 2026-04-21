import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import modopt as mo
from beam_jax import Beam, CSTube
from lifting_line_jax import LiftingLine
from optiplex import Plex, PlexT
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt


# aero setup
N = 31
b = 10.0
c_root = 1.0
c_tip = 0.65
lifting_line = LiftingLine(N, b, c_root, c_tip)

def aero_model(twist, rho_atm=1.225, v_inf=60):

    coef = lifting_line.solve_lifting_line_model(twist)
    CD = lifting_line.compute_drag(coef)
    lift_distribution = lifting_line.compute_lift_distribution(coef, rho_atm, v_inf)
    CL = lifting_line.compute_lift_coefficient(coef)

    lift = 0.5 * rho_atm * v_inf**2 * CL * lifting_line.S
    
    return CD, lift_distribution, lift

# structures setup
num_nodes   = 31
mesh        = np.zeros((num_nodes, 3))
mesh[:, 1]  = lifting_line.y
fixed_nodes = [num_nodes // 2] # the center node is fixed in all DOFs
r = 0.2 * (lifting_line.chord[:-1] + lifting_line.chord[1:]) / 4 # radius of the tube as a function of chord
m0 = 1e3
load_factor = 5
safety_factor = 3
tip_disp_target = 0.01

def structures_model(loads, thickness):

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 2].set(loads * load_factor * safety_factor)

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
# """
lw_scale = 1e-3
f_scale = 1e-2
disp_scale = 1e2
# lw_scale = 1e-4
# f_scale = 1e-2
# disp_scale = 1e1
# """
# lw_scale = 2#1
# f_scale = 1e-2
# disp_scale = 1e2

thickness0 = np.ones(num_nodes - 1) * 0.002
twist0 = np.ones(N) * np.deg2rad(5)

thickness_scale = 1e2
theta_scale = 1e1
f_copy_scale = 1e-1
scale = np.concatenate([np.ones(num_nodes - 1) * thickness_scale, np.ones(N) * theta_scale, np.ones(N) * f_copy_scale])

# run the aero model once to populate args
CD_init, lift_distribution_init, lift_init = aero_model(twist0)
# run the structures model once to populate args
right_tip_disp_init, left_tip_disp_init, weight_init = structures_model(lift_distribution_init, thickness0)

data = {'f_real': lift_distribution_init,
        'CD': CD_init,
        'Lift': lift_init,
        'right_tip_disp': right_tip_disp_init,
        'left_tip_disp': left_tip_disp_init,
        'Weight': weight_init,
        }

x_init = [twist0, thickness0, lift_distribution_init]

cd_history = []

def con(x):

    f_copy = x[2]

    # unpack the data dict
    f_real = data['f_real']
    lift = data['Lift']
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    # consensus constraint
    f_con = (f_copy - f_real) * f_scale

    # lift equals weight constraint
    l_equals_w = (lift - weight) * lw_scale
    # l_equals_w = (lift / weight - 1) * lw_scale

    # displacement constraints
    right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
    left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale

    return jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])


def aero_subproblem(x, y, mu):

    print('Solving aerodynamic subproblem...', end=' ', flush=True)

    twist = x[0]
    thickness = x[1]
    f_copy = x[2]
    v0 = twist
    
    # unpack the data dict
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    def jax_obj(v):

        # run the aero model
        CD, f_real, lift = aero_model(v)
        
        # compute the global constraints
        f_con = (f_copy - f_real) * f_scale
        l_equals_w = (lift - weight) * lw_scale
        # l_equals_w = (lift / weight - 1) * lw_scale
        right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
        left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale
        c = jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    x_scaler = np.ones(N) * 10 # twist scaler

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    twist_solution = optimizer.results['x'] / x_scaler

    # update the data dict
    CD, f_real, lift = aero_model(twist_solution)
    data['f_real'] = f_real
    data['CD'] = CD
    data['Lift'] = lift
    cd_history.append(CD)
    print('CD: ', CD, ' obj: ', optimizer.results['fun'])

    return [twist_solution, thickness, f_copy]

def struct_subproblem(x, y, mu):

    print('Solving structural subproblem....', end=' ', flush=True)

    twist = x[0]
    t = x[1]
    f_copy = x[2]
    v0 = np.concatenate([t, f_copy])
    
    # unpack the data dict
    f_real = data['f_real']
    CD = data['CD']
    lift = data['Lift']

    def jax_obj(v):

        t = v[:num_nodes - 1]
        f_copy = v[num_nodes - 1:]

        # run the structures model
        right_tip_disp, left_tip_disp, weight = structures_model(f_copy, t)

        # compute the global constraints
        f_con = (f_copy - f_real) * f_scale
        l_equals_w = (lift - weight) * lw_scale
        # l_equals_w = (lift / weight - 1) * lw_scale
        right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
        left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale
        c = jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    tl = np.ones(num_nodes - 1) * 0.001     # min gauge
    tu = np.ones(num_nodes - 1) * np.inf    # thickness upper
    fl = np.ones(num_nodes) * -np.inf       # f lower
    fu = np.ones(num_nodes) * np.inf        # f upper
    xl = np.concatenate([tl, fl])
    xu = np.concatenate([tu, fu])
    t_scaler = np.ones(num_nodes - 1) * 100 # thickness scaler
    l_scaler = np.ones(num_nodes) * 0.1       # f scaler
    x_scaler = np.concatenate([t_scaler, l_scaler])

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-8}, turn_off_outputs=True)
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



opt = PlexT(subproblems=[aero_subproblem, struct_subproblem],
           x_init=x_init,
           con=con,
           scale=scale,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=50,
          eps=1e-4,#1e-4, # 1e-3 # inner loop
          tol=1e-3,#0.5e-6,#1e-5,#1e-4, # 1e-4 # outer loop feasibility
          rho=1.5, # 1.2
          mu=3,#10
          max_mu=1e3,
          )

# works for mu = 1, 2,

print('Lagrange multipliers: ', opt.y)
solution = opt.x

twist = solution[0]
thickness = solution[1]
f_copy = solution[2]


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

f_real = data['f_real']
f_con = (f_copy - f_real) * f_scale
l_equals_w = (lift - weight) * lw_scale
right_disp_con = (right_tip_disp - tip_disp_target) * disp_scale
left_disp_con = (left_tip_disp - tip_disp_target) * disp_scale

print('Scaled constraint values: ')
print('f_con max abs: ', jnp.max(jnp.abs(f_con)))
print('l_equals_w: ', l_equals_w)
print('right_disp_con: ', right_disp_con)
print('left_disp_con: ', left_disp_con)


solution = np.load('examples/aero_structural/solution.npz')
x_star = np.concatenate([solution['twist'], solution['thickness']])

history_vecs = [np.concatenate(h[:2]) for h in opt.history]
error = [np.linalg.norm((x - x_star) / x_star) for x in history_vecs]

print('CD: ', cd_history[-1]) # correct value should be CD:  0.013790146790628004
# currently sometimes getting CD:  0.01691472981927389
# CD:  0.01691545746597533
# CD:  0.016894571235545806
# CD:  0.016419344396332646
# CD:  0.01687470315629312 with birkin scheme

# plt.semilogy(error)
# plt.xlabel('Iteration')
# plt.ylabel('Relative error')
# plt.show()

plt.semilogy(opt.x_time, error)
plt.xlabel('Time (s)')
plt.ylabel('Relative error')
plt.show()

plt.plot(opt.x_time, error)
plt.xlabel('Time (s)')
plt.ylabel('Relative error')
plt.show()

plt.semilogy(opt.x_time, opt.mu_history)
plt.xlabel('Time (s)')
plt.ylabel('Penalty parameter')
plt.show()

# plt.plot(solution['twist'], label='Reference twist')
# plt.plot(twist, label='Plex twist')
# plt.legend()
# plt.xlabel('Spanwise location')
# plt.ylabel('Twist (rad)')
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lifting_line.y, solution['twist'], label='Reference twist')
ax1.plot(lifting_line.y, twist, label='Plex twist')
ax1.set_xlabel('Spanwise location')
ax1.set_ylabel('Twist (rad)')
ax1.legend()
ax2.plot(solution['thickness'], label='Reference thickness')
ax2.plot(thickness, label='Plex thickness')
ax2.set_xlabel('Spanwise location')
ax2.set_ylabel('Thickness (m)')
ax2.legend()
plt.tight_layout()
plt.show()


f_real = data['f_real']
plt.plot(f_copy, label='f_copy')
plt.plot(f_real, label='f_real')
plt.legend()
plt.xlabel('Spanwise location')
plt.ylabel('Load (N)')
plt.show()

plt.plot(cd_history)
plt.xlabel('Iteration')
plt.ylabel('CD')
plt.show()

# save error history and mu history and x_time and mu_time
# np.savez('examples/aero_structural/history2_m2.npz', error=error, mu_history=opt.mu_history, x_time=opt.x_time)