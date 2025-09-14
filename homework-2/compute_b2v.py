import numpy as np
from scipy.integrate import trapezoid

def hard_sphere(r, sigma):
	return np.where(r < sigma, 1000, 0.0) # using 1000 as infinity, masking to create the function


def lennard_jones(r, epsilon=0.01, sigma=3.4):
	sr6 = (sigma / r) ** 6
	sr12 = sr6 ** 2
	return 4 * epsilon * (sr12 - sr6) # from previous homework


def square_well(r, lamb, epsilon, sigma):
	v = np.zeros(len(r))
	v[r < sigma] = 1000
	v[(r >= sigma) & (r < lamb * sigma)] = -epsilon # masking to create the function
	return v


def compute_b2v(T, model, **kwargs):
	
	r = np.linspace(0.001, 3.4*5, 1000)
	dr = r[1] - r[0]
	kb = 8.617e-5
	avagadro = 6.022e23
	first_part = np.exp(-model(r, *list(kwargs.values())) / (kb * T)) - 1
	second_part = r**2
	integrand = first_part * second_part
	return -2*np.pi*avagadro*trapezoid(y = integrand,x = r, dx=dr)*1e-24 # convert to cm^3/mol


print("Leonard-Jones at 100K:", compute_b2v(100, lennard_jones))
print("Square Well at 100K:",compute_b2v(100, square_well, sigma=3.4, epsilon=0.01, lamb = 1.5))
print("Hard Sphere at 100K:",compute_b2v(100, hard_sphere, sigma=3.4))

T_range = np.linspace(100, 800, 10)
b2v_lj = [compute_b2v(T, lennard_jones) for T in T_range]
b2v_sw = [compute_b2v(T, square_well, sigma=3.4, epsilon=0.01, lamb = 1.5) for T in T_range]
b2v_hs = [compute_b2v(T, hard_sphere, sigma=3.4) for T in T_range]

import matplotlib.pyplot as plt

plt.plot(T_range, b2v_lj, label='Lennard-Jones')
plt.plot(T_range, b2v_sw, label='Square Well')
plt.plot(T_range, b2v_hs, label='Hard Sphere')
plt.axhline(0, color='k', ls='--')
plt.xlabel('Temperature (K)')
plt.ylabel('$B_{2v}$ ($cm^3$/mol)')
plt.legend()
plt.savefig('./homework-2-2/b2v_comparison.png', dpi=300)

import pandas as pd

df = pd.DataFrame({'Temperature (K)': T_range,
'Lennard-Jones': b2v_lj,
'Square Well': b2v_sw,
'Hard Sphere': b2v_hs})
df.to_csv('./homework-2-2/b2v_values.csv', index=False)
