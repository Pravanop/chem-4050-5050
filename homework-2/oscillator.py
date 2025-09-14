import numpy as np
import matplotlib.pyplot as plt

h = 1
m = 1
omega = 1
D = 10
beta = np.sqrt(1 / (2 * D))
L = 40
N = 2000
x = np.linspace(-L / 2, L / 2, N)


def V_harmonic(x):
	return 0.5 * omega ** 2 * x ** 2


def V_morse(x, D, beta, x0):
	return D * (1.0 - np.exp(-beta * (x - x0))) ** 2 # x0 is 0 because we center the potential at 0

def pot_matrix(x, V):
	n = len(x)
	V_matrix = np.zeros((n, n))
	for i in range(n):
		V_matrix[i, i] = V[i]
	return V_matrix


laplace = -2 * np.diag(np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N - 1), -1)
laplace *= 1/(x[1] - x[0])**2


def build_H(x, V):
	return -h**2/(2*m) * laplace + pot_matrix(x, V)

H_harmonic = build_H(x, V_harmonic(x))
E, phi = np.linalg.eigh(H_harmonic)

E_sorted_arg = np.argsort(E)
E_sorted = E[E_sorted_arg] # sort the lowest to highest eigenvalues
phi_sorted = phi[:, E_sorted_arg]

for n in range(10):
	plt.plot(x, phi_sorted[:, n] + E_sorted[n], label=f'n={n}')
plt.ylim(0, 10)
plt.xlabel('x')
plt.ylabel('Energy')
plt.legend()
plt.title('Harmonic Oscillator Energy Levels')
plt.savefig('homework-2/homework-2-grad/harmonic_oscillator.png')
plt.show()

H_aharmonic = build_H(x, V_morse(x, D, beta, 0))
E, phi = np.linalg.eigh(H_aharmonic)

E_sorted_arg = np.argsort(E)
E_sorted = E[E_sorted_arg]
phi_sorted = phi[:, E_sorted_arg]

for n in range(10):
	plt.plot(x, phi_sorted[:, n] + E_sorted[n], label=f'n={n}')
plt.ylim(0, 9)
plt.xlabel('x')
plt.ylabel('Energy')
plt.legend()
plt.title('Aharmonic Oscillator Energy Levels')
plt.savefig('homework-2/homework-2-grad/aharmonic_oscillator.png')

