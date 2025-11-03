import numpy as np


def _mass(payload_mass, 
          not_wing_mass, 
          wing_area,
          wing_span,
          taper_ratio: float = 0.5,
          mean_aerodynamic_chord: float = 4.5,
          wing_sweep: float = np.deg2rad(35),
          n: float = 1.5 * 3.5,
          K: float = 0.003,
          density: float = 2711,
          t_over_c: float = 0.15
          ):

    aspect_ratio = wing_span**2 / wing_area

    wing_weight_newtons = (wing_area * mean_aerodynamic_chord * t_over_c * density * K *
                   (aspect_ratio * n / np.cos(wing_sweep))**0.6 *
                   taper_ratio**0.04 * 9.81)
    
    wing_mass = wing_weight_newtons / 9.81


    return payload_mass + wing_mass + not_wing_mass





if __name__ == "__main__":

    payload_mass = 10000 # (kg)

    not_wing_mass = 20000 # (kg)

    wing_span = 35.0 # (m)

    wing_area = 71 # (m^2)

    mass = _mass(payload_mass,
                 not_wing_mass,
                 wing_area,
                 wing_span,)
    

    print('Total Mass (kg): ', mass)