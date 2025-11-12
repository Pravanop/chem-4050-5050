import numpy as np


# using the pseudocode from the project description to implement the MC simulation

def initialize_lattice(size):
    lattice = np.zeros((size, size), dtype=int)
    return lattice


def compute_neighbor_indices(size):
    neighbor_indices = {}
    for x in range(size):
        for y in range(size):
            neighbors = [
                ((x - 1) % size, y),
                ((x + 1) % size, y),
                (x, (y - 1) % size),
                (x, (y + 1) % size),
            ]
            neighbor_indices[(x, y)] = neighbors
    return neighbor_indices


def calculate_interaction_energy(lattice, site, particle, neighbor_indices,
                                 epsilon_AA, epsilon_BB, epsilon_AB):
    x, y = site
    interaction_energy = 0.0
    for nx, ny in neighbor_indices[(x, y)]:
        neighbor_particle = lattice[nx, ny]
        if neighbor_particle == 0: # deviates from the pseudocode slightly to skip empty neighbors,
            # because I do not want too many nested ifs
            continue
            
        if particle == 1:
            if neighbor_particle == 1:
                interaction_energy += epsilon_AA
            else:
                interaction_energy += epsilon_AB
        else:
            if neighbor_particle == 2:
                interaction_energy += epsilon_BB
            else:
                interaction_energy += epsilon_AB
    return interaction_energy


def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params):
    size = lattice.shape[0]
    N_sites = size * size
    beta = 1.0 / params["T"]

    epsA = params["epsilon_A"]
    epsB = params["epsilon_B"]
    epsAA = params["epsilon_AA"]
    epsBB = params["epsilon_BB"]
    epsAB = params["epsilon_AB"]
    muA = params["mu_A"]
    muB = params["mu_B"]

    add_particle = np.random.random() < 0.5

    if add_particle:
        if N_empty == 0:
            return N_A, N_B, N_empty
        empties = np.argwhere(lattice == 0)
        sx, sy = empties[np.random.randint(len(empties))]
        add_A = np.random.random() < 0.5
        if add_A:
            particle = 1
            mu = muA
            eps = epsA
            N_s = N_A
        else:
            particle = 2
            mu = muB
            eps = epsB
            N_s = N_B

        delta_E = eps + calculate_interaction_energy(
            lattice, (sx, sy), particle, neighbor_indices, epsAA, epsBB, epsAB
        )

        acc_prob = min(1.0, (N_empty / (N_s + 1)) * np.exp(-beta * (delta_E - mu)))
        if np.random.random() < acc_prob:
            lattice[sx, sy] = particle
            if particle == 1:
                N_A += 1
            else:
                N_B += 1
            N_empty -= 1

    else:
        if N_sites - N_empty == 0:
            return N_A, N_B, N_empty
        occ = np.argwhere(lattice != 0)
        sx, sy = occ[np.random.randint(len(occ))]
        particle = lattice[sx, sy]
        if particle == 1:
            mu = muA
            eps = epsA
            N_s = N_A
        else:
            mu = muB
            eps = epsB
            N_s = N_B

        delta_E = -(eps + calculate_interaction_energy(
            lattice, (sx, sy), particle, neighbor_indices, epsAA, epsBB, epsAB
        ))

        acc_prob = min(1.0, (N_s / (N_empty + 1)) * np.exp(-beta * (delta_E + mu)))
        if np.random.random() < acc_prob:
            lattice[sx, sy] = 0
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1

    return N_A, N_B, N_empty


def run_simulation(size, n_steps, params):
    lattice = initialize_lattice(size)
    neighbor_indices = compute_neighbor_indices(size)
    N_sites = size * size
    N_A, N_B, N_empty = 0, 0, N_sites
    coverage_A = np.zeros(n_steps)
    coverage_B = np.zeros(n_steps)

    for step in range(n_steps):
        N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty,
                                         neighbor_indices, params)
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites

    return lattice, coverage_A, coverage_B


def plot_lattice(lattice, ax, title):
    size = lattice.shape[0]
    for x in range(size):
        for y in range(size):
            if lattice[x, y] == 1:
                ax.plot(x + 0.5, y + 0.5, 'o', color='red', markersize=12)
            elif lattice[x, y] == 2:
                ax.plot(x + 0.5, y + 0.5, 'o', color='blue', markersize=12)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(range(size+1), minor=True)
    ax.set_yticks(range(size+1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_title(title)
    return ax