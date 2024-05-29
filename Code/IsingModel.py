import numpy as np
from numba import jit
from HelperFunctions import Calculate_Energy, Calculate_Magnetism, Plot, Plot_save, Find_Images_Directory, Estimate_Largest_Cluster
from tqdm import tqdm
import os

@jit(nopython=True)
def MCMC_step(spin_lattice, temperature, B_field, size):
    for i in range(size):
        for j in range(size):
            spin = spin_lattice[i, j]
            neighbor_sum = (
                  spin_lattice[(i + 1) % size, j]
                + spin_lattice[i, (j + 1) % size]
                + spin_lattice[(i - 1) % size, j]
                + spin_lattice[i, (j - 1) % size]
            )

            energy_difference = 2 * spin * neighbor_sum + 2 * B_field * spin 

            if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / temperature):
                spin_lattice[i, j] *= -1

def Ising_Model_Simulation(size, temperature, steps, spin_lattice, B_field):
    magnetism = []
    for time_step in tqdm(range(steps), desc="Running Simulation"):
        MCMC_step(spin_lattice, temperature, B_field, size)
        magnetism.append(Calculate_Magnetism(spin_lattice))
    return spin_lattice, magnetism

def Run_Simulation_For_Temperatures(lattice_size, temperatures, steps, B_field):
    initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    plot = True
    images_directory = Find_Images_Directory()
    save_directory = os.path.join(images_directory, f"lattice_{lattice_size}")

    results = {}
    for temp in temperatures:
        lattice_copy = np.copy(initial_lattice)
        spin_lattice, magnetism = Ising_Model_Simulation(lattice_size, temp, steps, lattice_copy, B_field)
        largest_cluster_size = Estimate_Largest_Cluster(spin_lattice)
        
        results[temp] = {
            'final_lattice': spin_lattice,
            'magnetism': magnetism,
            'largest_cluster_size': largest_cluster_size
        }

        Plot(spin_lattice, f'Final Spin Configuration at T={temp}')
        Plot_save(spin_lattice, f'Final Spin Configuration, Lattice size: {lattice_size}, T={temp}', save_directory, f'final_steps{steps}_temp{temp}_B{B_field}.png')
        
        print(f'Temperature: {temp}, Largest Cluster Size: {largest_cluster_size}')
        
    return results

# Example usage
if __name__ == "__main__":
    lattice_size = 20
    temperatures = [10, 5, 4, 3, 2.5]
    steps_per_dipole = 100
    B_field = -1

    results = Run_Simulation_For_Temperatures(lattice_size, temperatures, steps_per_dipole, B_field)
