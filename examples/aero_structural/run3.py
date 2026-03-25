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


thickness0 = np.ones(num_nodes - 1) * 0.002
twist0 = np.ones(N) * np.deg2rad(5)
x_init = [twist0, thickness0]

# run the aero model once to populate args
CD_init, lift_distribution_init, lift_init = aero_model(twist0)
# run the structures model once to populate args
right_tip_disp_init, left_tip_disp_init, weight_init = structures_model(lift_distribution_init, thickness0)

data_init = {
            'Loads': lift_distribution_init,
            'CD': CD_init,
            'Lift': lift_init,
            'right_tip_disp': right_tip_disp_init,
            'left_tip_disp': left_tip_disp_init,
            'Weight': weight_init,
            }


def con(x, data):

    # unpack the data dict
    loads = data['Loads']
    CD = data['CD']
    lift = data['Lift']
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    con = jnp.zeros(3)
    con = con.at[0].set((left_tip_disp - tip_disp_target) * 1e2)  # equality constraint for now
    con = con.at[1].set((right_tip_disp - tip_disp_target) * 1e2) # equality constraint for now
    # con = con.at[2].set((lift - weight) * 1e-6)
    con = con.at[2].set((lift / weight) - 1)
    # con = con.at[2].set((lift - 0.5 * 1.225 * 60**2 * lifting_line.S * 0.5) * 1e-3)

    # print('cval: ', con)

    return con


def aero_subproblem(x, y, mu, data):
    data = data.copy()

    print('Solving aerodynamic subproblem...')

    twist = x[0]
    thickness = x[1]
    v0 = twist
    
    # unpack the data dict
    lift_distribution = data['Loads']
    CD = data['CD']
    lift = data['Lift']
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    def jax_obj(v):
        
        twist = v
        x_hat = [twist, thickness]

        CD, lift_distribution, lift = aero_model(twist)
        
        cdata = {'Loads': lift_distribution,
                 'CD': CD,
                 'Lift': lift,
                 'right_tip_disp': right_tip_disp,
                 'left_tip_disp': left_tip_disp,
                 'Weight': weight,
                 }
        c = con(x_hat, cdata)

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    x_scaler = np.ones(N) * 10 # twist scaler

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 200, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    twist_solution = optimizer.results['x'] / x_scaler

    # update the data dict
    CD, lift_distribution, lift = aero_model(twist_solution)
    data['Loads'] = lift_distribution
    data['CD'] = CD
    data['Lift'] = lift

    print('CD: ', CD)

    return [twist_solution, thickness], data

def struct_subproblem(x, y, mu, data):
    data = data.copy()

    print('Solving structural subproblem...')

    twist = x[0]
    thickness = x[1]
    v0 = thickness
    
    # unpack the data dict
    lift_distribution = data['Loads']
    CD = data['CD']
    lift = data['Lift']
    right_tip_disp = data['right_tip_disp']
    left_tip_disp = data['left_tip_disp']
    weight = data['Weight']

    def jax_obj(v):

        thickness = v
        x_hat = [twist, thickness]

        # update the data dict with the new thickness
        right_tip_disp, left_tip_disp, weight = structures_model(lift_distribution, thickness)

        cdata = {'Loads': lift_distribution,
                 'CD': CD,
                 'Lift': lift,
                 'right_tip_disp': right_tip_disp,
                 'left_tip_disp': left_tip_disp,
                 'Weight': weight,
                 }
        c = con(x_hat, cdata)

        return 1e3 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
    
    xl = np.ones(num_nodes - 1) * 0.001     # min gauge
    xu = np.ones(num_nodes - 1) * np.inf    # thickness upper
    x_scaler = np.ones(num_nodes - 1) * 100 # thickness scaler

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu, x_scaler=x_scaler)
    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 200, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    thickness_solution = optimizer.results['x'] / x_scaler

    # update the data dict
    right_tip_disp, left_tip_disp, weight = structures_model(lift_distribution, thickness_solution)
    data['right_tip_disp'] = right_tip_disp
    data['left_tip_disp'] = left_tip_disp
    data['Weight'] = weight

    return [twist, thickness_solution], data



opt = Plex(subproblems=[aero_subproblem, struct_subproblem],
           x_init=x_init,
           con=con,
           data=data_init,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          ATOL_out=1e-4, 
          RTOL_out=1e-4,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.2,
          mu=50.0,
          )

solution = opt.x

twist = solution[0]
thickness = solution[1]


plt.plot(lifting_line.y, twist)
plt.show()

plt.plot(thickness)
plt.show()