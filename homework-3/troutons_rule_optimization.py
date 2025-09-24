from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./trouton.csv')

H_v = df['H_v (kcal/mol)'].values
T_b = df['T_B (K)'].values


H_v_j = H_v * 4184  # Convert kcal/mol to J/mol

def objective_fn(params, T_b, H_v):
	slope, intercept = params
	H_v_pred = slope * T_b + intercept
	residuals = H_v - H_v_pred
	return sum(residuals**2)

initial_guess = [0.1, 0]

result = minimize(objective_fn, initial_guess, args=(T_b, H_v_j))
slope, intercept = result.x
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

plt.scatter(T_b, H_v_j, label='Data', color='blue')
plt.plot(T_b, slope * T_b + intercept, color='red', label='Fitted line')
plt.text(100, 250000, f'H$_v$ = {slope:.2f}*T$_b$ +  {intercept:.2f}')
plt.xlabel('$T_B$ (K)')
plt.ylabel('$H_v$ (J/mol)')
plt.title("Trouton’s Rule Optimization.")
plt.savefig('homework-3-2/troutons_rule_fit.png', dpi=300)

print("Comments on the comparison with the previous implementation: There is almost no difference in both the answers.")


