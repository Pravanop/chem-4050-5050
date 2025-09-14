import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def lennard_jones(r, epsilon=0.01, sigma=3.4):
	"""Compute Lennard-Jones potential."""
	sr6 = (sigma / r) ** 6
	sr12 = sr6 ** 2
	return 4 * epsilon * (sr12 - sr6)



optimize_distance = minimize(lennard_jones, 4.0)
r_minima = np.round(optimize_distance.x[0],3)  # to keep things clean
r_x = np.linspace(3, 6, 100)
e_x = lennard_jones(r_x)
plt.plot(r_x, e_x)
plt.axvline(r_minima, color='red', linestyle='--')
plt.ylabel('V(r) eV')
plt.xlabel('r (Å)')
plt.savefig('homework-2/homework-2-1/optimal_distance_dimer.png', dpi=300)

#copied from last homework
def calculate_bond_length(coord1, coord2):
	d = np.array(coord1) - np.array(coord2)
	d = d**2
	d = np.sqrt(np.sum(d))
	if d > 2:
		print("Warning: Bond length is greater than 2 Angstroms, which is too long.")

	return d

def calculate_bond_angle(coord1, coord2, coord3):
	ab = np.array(coord1) - np.array(coord2)
	bc = np.array(coord3) - np.array(coord2)
	dot_prod = np.dot(ab, bc)
	mag_ab = np.sqrt(np.sum(ab**2))
	mag_bc = np.sqrt(np.sum(bc**2))
	cos_theta = dot_prod / (mag_ab * mag_bc)
	theta = np.arccos(cos_theta)
	if theta < np.pi:
		angle = 'acute'
	elif theta == np.pi:
		angle = 'right'
	else:
		angle = 'obtuse'

	return theta, angle

first_atom_coords = [0.0, 0.0]
second_atom_coords = [0.0, r_minima]
bond_length = calculate_bond_length(first_atom_coords, second_atom_coords)
print(f"Minimum distance: {r_minima} Å")
print(f"Optimal bond length: {bond_length} Å")
print("Bond angle = 180 degrees (dimer)")

def write_xyz(filename, coords, atoms):
	
	file_list = []
	file_list.append(f"{len(atoms)}\n")
	file_list.append("Argon dimer\n")
	for atom, coord in zip(atoms, coords):
		file_list.append(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")
	
	with open(filename, 'w') as f:
		f.writelines(file_list)

atoms = ['Ar', 'Ar']
coords = [first_atom_coords + [0.0], second_atom_coords + [0.0]] # include z coordinates
write_xyz('homework-2/homework-2-1/argon_dimer.xyz', coords, atoms)

