import numpy as np
import matplotlib.pyplot as plt
from MC_code import run_simulation, plot_lattice

#packing the example code as a function to run different scenarios

k_B = 8.617333262145e-5  # eV/K
def run_project(mus_A, Ts, n_steps, params, size, scenario_name):
	# Run the simulation
	np.random.seed(7)
	final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
	mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
	mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
	for i, param in enumerate(params):
		lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
		final_lattice[i // len(Ts), i % len(Ts)] = lattice
		mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
		mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])
	
	# Plot the T-mu_A phase diagram
	fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(10.5, 8.5))
	
	# Mean coverage of A
	axs[0].pcolormesh(mus_A, Ts, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
	axs[0].set_title(r'$\langle \theta_H \rangle$')
	axs[0].set_xlabel(r'$\mu_H$ (eV)')
	axs[0].set_ylabel(r'$T*$')
	
	# Mean coverage of B
	axs[1].pcolormesh(mus_A, Ts, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
	axs[1].set_title(r'$\langle \theta_N \rangle$')
	axs[1].set_xlabel(r'$\mu_H$ (eV)')
	axs[1].set_yticks([])
	
	# Mean total coverage
	cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
	axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
	axs[2].set_xlabel(r'$\mu_H$ (eV)')
	axs[2].set_yticks([])
	fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)
	
	# Plot the final lattice configuration
	
	# mu_A = -0.2 eV and T = 0.01 / k
	axs[3] = plot_lattice(final_lattice[0, 3], axs[3], r'$\mu_H = -0.2$ eV, $T = 0.01 / k$')
	
	# mu_A = -0.1 eV and T = 0.01 / k
	axs[4] = plot_lattice(final_lattice[3, 3], axs[4], r'$\mu_H = -0.1$ eV, $T = 0.01 / k$')
	
	# mu_A = 0 eV and T = 0.01 / k
	axs[5] = plot_lattice(final_lattice[6, 3], axs[5], r'$\mu_H = 0$ eV, $T = 0.01 / k$')
	
	plt.tight_layout()
	plt.savefig(f'./plots/{scenario_name}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
	scenarios = {
		"Ideal Mixture": {
			"epsilon_A": -0.1,
			"epsilon_B": -0.1,
			"epsilon_AA": 0.0,
			"epsilon_BB": 0.0,
			"epsilon_AB": 0.0,
		},
		"Repulsive": {
			"epsilon_A": -0.1,
			"epsilon_B": -0.1,
			"epsilon_AA": 0.05,
			"epsilon_BB": 0.05,
			"epsilon_AB": 0.05,
		},
		"Attractive": {
			"epsilon_A": -0.1,
			"epsilon_B": -0.1,
			"epsilon_AA": -0.05,
			"epsilon_BB": -0.05,
			"epsilon_AB": -0.05,
		},
		"Immiscible": {
			"epsilon_A": -0.1,
			"epsilon_B": -0.1,
			"epsilon_AA": -0.05,
			"epsilon_BB": -0.05,
			"epsilon_AB": 0.05,
		},
		"Like Dissolves Unlike": {
			"epsilon_A": -0.1,
			"epsilon_B": -0.1,
			"epsilon_AA": 0.05,
			"epsilon_BB": 0.05,
			"epsilon_AB": -0.05,
		},
	}
	
	size = 4
	n_steps = 10000
	mus_A = np.linspace(-0.2, 0, 7)
	
	Ts = np.linspace(0.001, 0.049, 7)
	
	for scenario_name, scenario_params in scenarios.items():
		params = []
		for mu_A in mus_A:
			for T in Ts:
				scenario_params.update({
					'mu_A': mu_A,
					'mu_B': -0.1,
					'T': T  # Temperature (in units of kb)
				})
				params.append(scenario_params)
		run_project(mus_A, Ts, n_steps, params, size, scenario_name)
		print("Completed scenario:", scenario_name)
