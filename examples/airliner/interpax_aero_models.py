import interpax
import numpy as np

_mach = np.linspace(0, 1, 5)
_CL_alpha_data = np.array([5.2, 5.1, 5.1, 5.4, 7.1])
_CD_0_data = np.array([1.8, 1.8, 1.8, 1.8, 2.5]) * 1e-2
_K_data = np.array([0.55, 0.55, 0.55, 0.6, 1.7]) / 10

_interp_K = interpax.Akima1DInterpolator(_mach, _K_data, check=False, extrapolate=True)
_interp_CL_alpha = interpax.Interpolator1D(_mach, _CL_alpha_data, method='cubic2', extrap=True)
_interp_CD_0 = interpax.Akima1DInterpolator(_mach, _CD_0_data, check=False, extrapolate=True)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _mach_test = np.linspace(0, 1, 100)
    _CL_alpha_test = _interp_CL_alpha(_mach_test)
    _CD_0_test = _interp_CD_0(_mach_test)
    _K_test = _interp_K(_mach_test)

    plt.plot(_mach_test, _CL_alpha_test, label="CL_alpha")
    plt.plot(_mach_test, _CD_0_test * 1e2, label="CD_0")
    plt.plot(_mach_test, _K_test * 10, label="K")
    plt.scatter(_mach, _CL_alpha_data)
    plt.scatter(_mach, _CD_0_data * 1e2)
    plt.scatter(_mach, _K_data * 10)
    plt.legend()
    plt.show()