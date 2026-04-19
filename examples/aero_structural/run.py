import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import modopt as mo
from beam_jax import Beam, CSTube
from lifting_line_jax import LiftingLine
from optiplex import Plex
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
lw_scale = 1e-3
f_scale = 1
disp_scale = 1e1

thickness0 = np.ones(num_nodes - 1) * 0.002
twist0 = np.ones(N) * np.deg2rad(5)
x_init = [twist0, thickness0]

# run the aero model once to populate args
CD_init, f_init, lift_init = aero_model(twist0)

data = {'f_real': f_init, 'CD': CD_init, 'Lift': lift_init}



def con(x):

    twist = x[0]
    thickness = x[1]

    CD, f, lift = aero_model(twist)

    right_tip_disp, left_tip_disp, weight = structures_model(f, thickness)

    con = jnp.zeros(3)
    con = con.at[0].set((left_tip_disp - tip_disp_target) * disp_scale)  # equality constraint for now
    con = con.at[1].set((right_tip_disp - tip_disp_target) * disp_scale) # equality constraint for now
    con = con.at[2].set((lift - weight) * lw_scale)

    return con


def aero_subproblem(x, y, mu):

    print('Solving aerodynamic subproblem...')

    twist = x[0]
    thickness = x[1]

    def jax_obj(v):
        
        twist = v
        x[0] = twist

        CD, f, lift = aero_model(twist)

        right_tip_disp, left_tip_disp, weight = structures_model(f, thickness)

        c = jnp.zeros(3)
        c = c.at[0].set((left_tip_disp - tip_disp_target) * disp_scale)  # equality constraint for now
        c = c.at[1].set((right_tip_disp - tip_disp_target) * disp_scale) # equality constraint for now
        c = c.at[2].set((lift - weight) * lw_scale)

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)

    jaxprob = mo.JaxProblem(x0=twist, jax_obj=jax_obj, x_scaler=10)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    twist_solution = optimizer.results['x'] / 10

    # update args in the data dict
    CD, f, lift = aero_model(twist_solution)
    data['CD'] = CD
    data['Lift'] = lift
    data['f_real'] = f

    return [twist_solution, thickness]

def struct_subproblem(x, y, mu):

    print('Solving structural subproblem...')

    twist = x[0]
    thickness = x[1]
    
    CD = data['CD']
    lift = data['Lift']
    f = data['f_real']

    def jax_obj(v):

        thickness = v
        x[1] = thickness

        right_tip_disp, left_tip_disp, weight = structures_model(f, thickness)

        c = jnp.zeros(3)
        c = c.at[0].set((left_tip_disp - tip_disp_target) * disp_scale)  # equality constraint for now
        c = c.at[1].set((right_tip_disp - tip_disp_target) * disp_scale) # equality constraint for now
        c = c.at[2].set((lift - weight) * lw_scale)

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)

    jaxprob = mo.JaxProblem(x0=thickness, jax_obj=jax_obj, xl=0.001, xu=np.inf, x_scaler=100)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    thickness_solution = optimizer.results['x'] / 100

    return [twist, thickness_solution]



opt = Plex(subproblems=[aero_subproblem, struct_subproblem],
           x_init=x_init,
           con=con,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-3, 
          RTOL_in=1e-3,
          ATOL_feas=1e-5,
          rho=1.2,
          mu=10.0,
          )

solution = opt.x

twist = solution[0]
thickness = solution[1]


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lifting_line.y, twist)
ax2.plot(thickness)
plt.tight_layout()
plt.show()




solution = np.load('examples/aero_structural/solution.npz')
x_star = np.concatenate([solution['twist'], solution['thickness']])

history_vecs = [np.concatenate(h[:2]) for h in opt.history]
error = [np.linalg.norm((x - x_star) / x_star) for x in history_vecs]

# plt.semilogy(error)
# plt.xlabel('Iteration')
# plt.ylabel('Relative error')
# plt.show()

plt.semilogy(opt.x_time, error)
plt.xlabel('Time (s)')
plt.ylabel('Relative error')
plt.show()

plt.semilogy(opt.x_time, opt.mu_history)
plt.xlabel('Time (s)')
plt.ylabel('Penalty parameter')
plt.show()

plt.plot(solution['twist'], label='Reference twist')
plt.plot(twist, label='Plex twist')
plt.legend()
plt.xlabel('Spanwise location')
plt.ylabel('Twist (rad)')
plt.show()