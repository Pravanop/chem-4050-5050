import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("internal_energy_and_heat_capacity_vs_T.csv", delimiter=",", skiprows=1)
T_vals, U_vals, Cv_vals = data[:, 0], data[:, 1], data[:, 2]

Cv_vals = Cv_vals/1.602e-19 #eV to J
idx_max = np.argmax(Cv_vals)
T_diss = T_vals[idx_max]
Cv_max = Cv_vals[idx_max]

# Plot
plt.figure(figsize=(6,4))
plt.plot(T_vals, Cv_vals, lw=2, label=r"$C_V(T)$")
plt.axvline(T_diss, color='r', ls='--', label=r"$T_{diss} = %.1f K$" % T_diss)
plt.xlabel("Temperature (K)", fontsize=12)
plt.ylabel(r"Heat Capacity $C_V$ (J/K)", fontsize=12)
plt.title("Heat Capacity vs Temperature for LJ Dimer")
plt.grid(False)
plt.tight_layout()
plt.savefig("heat_capacity_vs_temperature.png", dpi=300)
