import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


df = pd.read_csv('./trouton.csv')

H_v = df['H_v (kcal/mol)'].values
T_b = df['T_B (K)'].values


# Taken from Lecture 7
def ols_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator


def ols_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean


def ols(x, y):
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept


slope, intercept = ols(T_b, H_v)
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

#kcal to J
slope_j = slope * 4184 # J/(mol·K)

troutons_value = 88   # Assuming standard
print(f"Trouton's rule value: {troutons_value} J/(mol·K)")
print(f"Slope in J/(mol·K): {slope_j} J/(mol·K)")

# it is around 20 J/(mol·K) off from Trouton's rule,
# which is a reasonable deviation considering simplicity of the model.

T_x = np.linspace(0, 2500, 100)
h_pred = slope * T_x + intercept

# predicting 95% confidence interval for slope and intercept

ssr = np.sum((H_v - (slope * T_b + intercept))**2)/ (len(H_v)-2)
slope_se = np.sqrt(ssr / np.sum((T_b - np.mean(T_b))**2)) # using eqns from lecture 8
intercept_se = np.sqrt(ssr * (1/len(H_v) + (np.mean(T_b)**2 / np.sum((T_b - np.mean(T_b))**2))))

t_value = t.ppf(1 - 0.95/2, df=len(H_v)-2)
slope_ci = slope_se * t_value
intercept_ci = intercept_se * t_value


plt.scatter(T_b, H_v*4184, label='Data Points', c = 'blue', edgecolors='k')
plt.plot(T_x, h_pred*4184, color='black', label='OLS Fit')
plt.text(100, 250000, f'H$_v$ = ({slope_j:.2f} +-{slope_ci*4184:.2f})*T$_b$ +  ({intercept*4184:.2f} +- {intercept_ci*4184:.2f})')
plt.xlabel('Temperature (K)')
plt.ylabel('H (J/mol)')
plt.savefig('homework-3-1/troutons_rule.png', dpi=300)


