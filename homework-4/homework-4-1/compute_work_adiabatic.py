import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

def compute_work_adiabatic(n, T, V_initial, V_final, gamma):
	
	R = 8.314  # J/(mol*K), universal gas constant
	if V_final == V_initial:
		return 0.0  # No work done if volumes are the same
	
	V = np.linspace(V_initial, V_final, 1000)
	P_initial = n*R*T/V_initial
	const = P_initial * V_initial**(gamma)
	P = const / (V*gamma)
	W = -trapezoid(P, V)
	return W

#table 1 parameters

n = 1.0  # moles
T = 300.0  # Kelvin
V_initial = 0.1  # m^3
gamma = 1.4
Vf_values = np.linspace(V_initial, 3*V_initial, 10)  # Final volumes from 0.1 to 1.0 m^3
y_values = [compute_work_adiabatic(n, T, V_initial, Vf, gamma) for Vf in Vf_values]
df = pd.DataFrame({'Vf (m^3)': Vf_values, 'Work (J)': y_values})
df.to_csv('work_adiabatic_data.csv')