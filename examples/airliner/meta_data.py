from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Data:
    eta: np.ndarray
    theta: np.ndarray
    tf: np.ndarray
    S: np.ndarray
    AR: np.ndarray
    fuel: np.ndarray
    v0: float
    nu: int
    nt: int


class NearestKeyDict(dict):
    """
    Dictionary that returns the value for the nearest key when exact match not found.
    """

    def __getitem__(self, key):
        # Try exact match first
        if key in self:
            return super().__getitem__(key)

        # Find nearest key
        keys = list(self.keys())
        if not keys:
            raise KeyError(f"Dictionary is empty")

        nearest_key = min(keys, key=lambda k: abs(k - key))
        return super().__getitem__(nearest_key)


Params: Dict[float, Data] = NearestKeyDict()
# Params: Dict[float, Data] = Data()

nu_6e6 = 300
Params[6e6] = Data(eta=np.linspace(0.7, 0.5, nu_6e6),
                   theta=np.linspace(np.deg2rad(7), np.deg2rad(5), nu_6e6),
                   tf=np.array([30000.0]),
                   S=np.array([90.0]),
                   AR=np.array([18.0]),
                   fuel=np.array([5000.0]),
                   v0=150.0,
                   nu=nu_6e6,
                   nt=10000,
                   )

nu_5e6 = 300
Params[5e6] = Data(eta=np.linspace(0.6, 0.4, nu_5e6),
                   theta=np.linspace(np.deg2rad(7), np.deg2rad(5), nu_5e6),
                   tf=np.array([25000.0]),
                   S=np.array([70.0]),
                   AR=np.array([23.0]),
                   fuel=np.array([6000.0]),
                   v0=150.0,
                   nu=nu_5e6,
                   nt=10000,
                   )

nu_4e6 = 200
Params[4e6] = Data(eta=np.linspace(0.6, 0.5, nu_4e6),
                   theta=np.linspace(np.deg2rad(6), np.deg2rad(6), nu_4e6),
                   tf=np.array([23000.0]),
                   S=np.array([75.0]),
                   AR=np.array([23.0]),
                   fuel=np.array([5000.0]),
                   v0=150.0,
                   nu=nu_4e6,
                   nt=10000,
                   )

nu_3e6 = 200
Params[3e6] = Data(eta=np.linspace(0.5, 0.4, nu_3e6),
                   theta=np.linspace(np.deg2rad(5), np.deg2rad(5), nu_3e6),
                   tf=np.array([16750.0]),
                   S=np.array([73.0]),
                   AR=np.array([22.0]),
                   fuel=np.array([4000.0]),
                   v0=150.0,
                   nu=nu_3e6,
                   nt=8000,
                   )

nu_2e6 = 200
Params[2e6] = Data(eta=np.linspace(0.5, 0.3, nu_2e6),
                   theta=np.linspace(np.deg2rad(5), np.deg2rad(4), nu_2e6),
                   tf=np.array([12000.0]),
                   S=np.array([80.0]),
                   AR=np.array([22.0]),
                   fuel=np.array([1930.0]),
                   v0=150.0,
                   nu=nu_2e6,
                   nt=9000,
                   )

nu_1e6 = 200
Params[1e6] = Data(eta=np.linspace(0.5, 0.3, nu_1e6),
                   theta=np.linspace(np.deg2rad(5), np.deg2rad(4), nu_1e6),
                   tf=np.array([10000.0]),
                   S=np.array([75.0]),
                   AR=np.array([22.0]),
                   fuel=np.array([1500.0]),
                   v0=160.0,
                   nu=nu_1e6,
                   nt=7000,
                   )