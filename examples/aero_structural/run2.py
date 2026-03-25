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
v_inf = 60
rho_atm = 1.225
q = 0.5 * rho_atm * v_inf**2
lifting_line = LiftingLine(N, b, c_root, c_tip)

def aero_model(twist):

    CD = lifting_line.compute_drag(twist)
    lift_distribution = lifting_line.compute_lift_distribution(twist, rho_atm, v_inf)
    CL = lifting_line.compute_lift_coefficient(twist)
    lift = CL * q * lifting_line.S
    
    return CD, lift_distribution, lift

# structures setup
num_nodes   = 31
mesh        = np.zeros((num_nodes, 3))
mesh[:, 1]  = lifting_line.y
fixed_nodes = [num_nodes // 2] # the center node is fixed in all DOFs
r = 0.2 * (lifting_line.chord[:-1] + lifting_line.chord[1:]) / 4 # radius of the tube as a function of chord
E = 69e9
G = 26e9
rho_mat = 3000
m0 = 1e3
load_factor = 5
safety_factor = 3
tip_disp_target = 0.01

def structures_model(loads, thickness):

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 2].set(loads * load_factor * safety_factor)

    cs = CSTube(radius=r, thickness=thickness)
    beam = Beam(mesh=mesh, E=E, G=G, rho=rho_mat,
                A=cs.area, J=cs.J, Iy=cs.Iy, Iz=cs.Iz, F=F, 
                fixed_nodes=fixed_nodes)
    u = beam.solve()
    u = jnp.linalg.norm(u[:, :3], axis=1)
    right_tip_disp, left_tip_disp = u[-1], u[0]

    mass = beam.mass + m0
    weight = mass * 9.81

    return right_tip_disp, left_tip_disp, weight


thickness0 = np.ones(num_nodes - 1) * 0.002
twist0 = np.ones(N) * np.deg2rad(5)
x_init = [twist0, thickness0]

# populate args 
# run the aero model once to populate args
CD_init, lift_distribution_init, lift_init = aero_model(twist0)
# run the structures model once to populate args
right_tip_disp, left_tip_disp, weight = structures_model(lift_distribution_init, thickness0)

args = {
        'Loads': lift_distribution_init,
        'CD': CD_init,
        'Lift': lift_init,
        'right_tip_disp': right_tip_disp,
        'left_tip_disp': left_tip_disp,
        'weight': weight,
        }



def global_constraints(x):

    twist = x[0]
    thickness = x[1]
    # slack = x[2]

    loads = args['Loads']

    right_tip_disp, left_tip_disp, weight = structures_model(loads, thickness)
    # right_tip_disp = args['right_tip_disp']
    # left_tip_disp = args['left_tip_disp']
    # weight = args['weight']

    lift = args['Lift']

    con = jnp.zeros(3)
    con = con.at[0].set((left_tip_disp - tip_disp_target) * 1e1)  # equality constraint for now
    con = con.at[1].set((right_tip_disp - tip_disp_target) * 1e1) # equality constraint for now
    con = con.at[2].set((lift - weight) * 1e-2)

    return con


def aero_subproblem(x, y, mu):

    print('Solving aerodynamic subproblem...')

    twist = x[0]
    thickness = x[1]
    # slack = x[2]

    def jax_obj(v):
        
        twist = v
        x[0] = twist

        CD = lifting_line.compute_drag(twist)
        lift_distribution = lifting_line.compute_lift_distribution(twist, rho_atm, v_inf)
        CL = lifting_line.compute_lift_coefficient(twist)
        lift = CL * q * lifting_line.S
        args['Loads'] = lift_distribution
        args['CD'] = CD
        args['Lift'] = lift

        c = global_constraints(x)

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    x0 = twist

    jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 200, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    twist_solution = optimizer.results['x']

    x[0] = twist_solution

    # update args
    lift_distribution = lifting_line.compute_lift_distribution(twist_solution, rho_atm, v_inf)
    CD = lifting_line.compute_drag(twist_solution)
    CL = lifting_line.compute_lift_coefficient(twist_solution)
    lift = CL * q * lifting_line.S
    args['Loads'] = lift_distribution
    args['CD'] = CD
    args['Lift'] = lift

    return x

def struct_subproblem(x, y, mu):

    print('Solving structural subproblem...')

    twist = x[0]
    thickness = x[1]
    # slack = x[2]

    loads = args['Loads']

    def jax_obj(v):

        thickness = v
        x[1] = thickness

        # update args for the new thickness
        right_tip_disp, left_tip_disp, weight = structures_model(loads, thickness)
        args['right_tip_disp'] = right_tip_disp
        args['left_tip_disp'] = left_tip_disp
        args['weight'] = weight

        c = global_constraints(x)

        CD = args['CD']

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    xl = np.ones(num_nodes - 1) * 0.001     # min gauge
    xu = np.ones(num_nodes - 1) * np.inf    # thickness upper
    x_scaler = np.ones(num_nodes - 1) * 100 # thickness scaler

    x0 = thickness

    jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=xl, xu=xu, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 200, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    thickness_solution = optimizer.results['x'] / x_scaler

    # update variables
    x[1] = thickness_solution

    # update args
    right_tip_disp, left_tip_disp, weight = structures_model(loads, thickness_solution)
    args['right_tip_disp'] = right_tip_disp
    args['left_tip_disp'] = left_tip_disp
    args['weight'] = weight

    return x

# def slack_update(x, y, mu):
#     print('Updating slack variables...')

#     con = global_constraints(x)
#     ineq_con = con[:2]
#     sigma = y[:2] # Lagrange multipliers for the inequality constraints

#     new_slack_variables = jnp.maximum(np.zeros((2)), -ineq_con - sigma / mu)
#     print('New slack variables: ', new_slack_variables)

#     x[2] = new_slack_variables

#     return x



opt = Plex(subproblems=[aero_subproblem, struct_subproblem],
           x_init=x_init,
           con=global_constraints,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=3,
          ATOL_out=1e-4, 
          RTOL_out=1e-4,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-5,
          rho=1.2,
          mu=10.0,
          )

solution = opt.x

twist = solution[0]
thickness = solution[1]


plt.plot(lifting_line.y, twist)
plt.show()

plt.plot(thickness)
plt.show()