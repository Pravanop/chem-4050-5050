import numpy as np
from scipy.constants import k

# Load partition function data
data = np.loadtxt("partition_function_vs_T.csv", delimiter=",", skiprows=1)
T_vals, Z_vals = data[:, 0], data[:, 1]
lnZ = np.log(Z_vals)
dlnZ_dT = np.gradient(lnZ, T_vals)
U = k * T_vals**2 * dlnZ_dT
Cv = np.gradient(U, T_vals)

np.savetxt(
	"internal_energy_and_heat_capacity_vs_T.csv",
    np.column_stack((T_vals, U, Cv)),
    delimiter=",",
    header="Temperature,U , Cv"
)
print("Saved: internal_energy_and_heat_capacity_vs_T.csv")