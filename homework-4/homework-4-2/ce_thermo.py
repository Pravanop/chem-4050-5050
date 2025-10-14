import numpy as np
from scipy.constants import k
import pandas as pd
import matplotlib.pyplot as plt

k_b = k * 6.242e18  # eV/K
def partition_function(Ei, gi, T):
	beta = 1/(k_b*T)
	print(beta)
	Z = [np.sum(gi*np.exp(-i*Ei), axis = 0) for i in beta]
	# Z = np.sum(gi*np.exp(-beta*Ei), axis = 0)
	return Z
	
	
def internal_energy(Ei, gi, T):
	beta = 1/(k_b*T)
	Z = partition_function(Ei, gi, T)
	U = np.gradient(np.log(Z), beta)
	return -U


def free_energy(Ei, gi, T):
	Z = partition_function(Ei, gi, T)
	F = -k_b*T * np.log(Z)
	return F


def entropy(F, T):
	return -np.gradient(F, T)


energies = {
	"degenerate": [[0.0], [14]],
	"soc": [[0.0, 0.28], [6, 8]],
	"soc_cfs": [[0.0, 0.12, 0.25, 0.32, 0.46], [4, 2, 2, 4, 2]]
}
T_list = np.linspace(300, 2000, 10)

fig, ax = plt.subplots(4, 1, figsize=(8, 8))
count = 0
for key, value in energies.items():
	Ei, gi = np.array(value[0]), np.array(value[1])
	
	Z = partition_function(Ei, gi, T_list)
	U = internal_energy(Ei, gi, T_list)
	F = free_energy(Ei, gi, T_list)
	S = entropy(F, T_list)
	df = pd.DataFrame({
		'T (K)': T_list,
		'Z': Z,
		'U (eV/mol)': U,
		'F (eV/mol)': F,
		'S (eV/mol/K)': S
	})
	ax[0].plot(T_list, Z, label=key)
	ax[1].plot(T_list, U, label=key)
	ax[2].plot(T_list, F, label=key)
	ax[3].plot(T_list, S, label=key)
	

	df.to_csv("ce_thermo_" + key + ".csv", index=False)
	count += 1
	
ax[0].set_ylabel('Z')
ax[1].set_ylabel('U')
ax[2].set_ylabel('F')
ax[3].set_ylabel('S')
ax[3].set_xlabel('T (K)')
ax[0].legend()
plt.subplots_adjust(hspace=0.0, wspace=0.0)
plt.savefig("ce_thermo.png")