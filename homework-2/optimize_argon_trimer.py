import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def lennard_jones(r, epsilon=0.01, sigma=3.4):
	"""Compute Lennard-Jones potential."""
	sr6 = (sigma / r) ** 6
	sr12 = sr6 ** 2
	return 4 * epsilon * (sr12 - sr6)


def lennard_jones_total(thetas):
	r12, x3, y3 = thetas  # unpack inputs
	r13 = np.sqrt(x3 ** 2 + y3 ** 2)
	r23 = np.sqrt((x3 - r12) ** 2 + y3 ** 2)  # give inputs that need to be minimized and derive distances manually
	return lennard_jones(r12) + lennard_jones(r13) + lennard_jones(r23)


optimize_distance = minimize(lennard_jones_total, x0=np.array([4.0, 6.0, 2.0]))

minima = np.round(optimize_distance.x,3)
minimum = lennard_jones_total(minima)
print(minima, minimum)

print(f"Minimum energy : {minimum} eV")
# e_minima =

#copied from last homework
def calculate_bond_length(coord1, coord2):
	d = np.array(coord1) - np.array(coord2)
	d = d ** 2
	d = np.sqrt(np.sum(d))
	if d > 2:
		print("Warning: Bond length is greater than 2 Angstroms, which is too long.")
	
	return d


def calculate_bond_angle(coord1, coord2, coord3):
	ab = np.array(coord1) - np.array(coord2)
	bc = np.array(coord3) - np.array(coord2)
	dot_prod = np.dot(ab, bc)
	mag_ab = np.sqrt(np.sum(ab ** 2))
	mag_bc = np.sqrt(np.sum(bc ** 2))
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
second_atom_coords = [0.0, minima[0]]
third_atom_coords = [minima[1], minima[2]]
bond_length1 = np.round(calculate_bond_length(first_atom_coords, second_atom_coords),3)
bond_length2 = np.round(calculate_bond_length(first_atom_coords, second_atom_coords),3)
bond_length3 = np.round(calculate_bond_length(second_atom_coords, third_atom_coords),3)
bond_angle, _ = calculate_bond_angle(first_atom_coords, second_atom_coords, third_atom_coords)
print(f"Optimal bond lengths: \n"
	  f"r12 =  {bond_length1} Å \n"
	  f"r13 =  {bond_length2} Å \n"
	  f"r23 =  {bond_length3} Å")
print(f"Bond angle = {np.round(np.rad2deg(bond_angle),3)} degrees")

def write_xyz(filename, coords, atoms):
	file_list = []
	file_list.append(f"{len(atoms)}\n")
	file_list.append("Argon dimer\n")
	for atom, coord in zip(atoms, coords):
		file_list.append(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")
	
	with open(filename, 'w') as f:
		f.writelines(file_list)

atoms = ['Ar', 'Ar', 'Ar']
coords = [first_atom_coords + [0.0], second_atom_coords + [0.0], third_atom_coords + [0.0]] # include z coordinates
write_xyz('./homework-2-1/argon_trimer.xyz', coords, atoms)
