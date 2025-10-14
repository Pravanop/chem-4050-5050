import numpy as np
from scipy.integrate import trapezoid
from scipy.constants import k, h
import pandas as pd

epsilon = 0.0103        # eV
sigma = 3.4             # Angstrom
V_box = 1e3             # Angstrom^3
Tmin, Tmax = 10, 1000   # K
nT = 20                 # number of temperature points
T_vals = np.linspace(Tmin, Tmax, nT)
kb = 8.617333262145e-5  # eV/K
eV_to_J = 1.602176634e-19
a_to_m = 1e-10

eps_J = epsilon
sig_m = sigma
V_m3 = V_box

# Physical constants for momentum prefactor
m_u = 1.66053906660e-27
M_Ar = 39.948
L_box_m = V_m3 ** (1/3)
hev = 4.1357e-15

def thermal_wavelength(T):
    beta = 1 / (k * T)
    return np.sqrt(beta * hev**2 / (2 * np.pi * M_Ar))

def V_LJ(r):
    return 4 * eps_J * ((sig_m / r)**12 - (sig_m / r)**6)

#the 6 dimensional integral over coordinates is reduced 1d as inter-atomic distance r
def compute_partition_function(T):
    beta = 1 / (kb * T)
    r_min = 1e-5
    r_max = 0.5 * L_box_m
    r_vals = np.linspace(r_min, r_max, 500)
    integrand = 4 * np.pi * r_vals**2 * np.exp(-beta * V_LJ(r_vals))
    radial_int = trapezoid(integrand, r_vals)
    Z_config = V_m3 * radial_int
    lam = thermal_wavelength(T)
    Z_full = Z_config / (hev**6 * lam**6)
    return Z_full

Z_vals = np.array([compute_partition_function(T) for T in T_vals])
print(Z_vals)
df = pd.DataFrame({"Temperature(K)": T_vals, "PartitionFunction": Z_vals})
df.to_csv("partition_function_vs_T.csv", index=False)

