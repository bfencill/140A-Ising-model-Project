import numpy as np
from numba import jit
from HelperFunctions import Calculate_Energy, Calculate_Magnetism, Plot, Plot_save, Find_Images_Directory
from tqdm import tqdm
import os

@jit(nopython=True)
def MCMC_step(spin_lattice, temperature, B_field, size, plot, time_step):
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

def Ising_Model_Simulation(size, temperature, steps, spin_lattice, B_field, plot):
    magnetism = []
    for time_step in tqdm(range(steps) , desc= "Running Simulation"):
        MCMC_step(spin_lattice, temperature, B_field, size, plot, time_step)
        magnetism.append(Calculate_Magnetism(spin_lattice))
    return spin_lattice, magnetism



# Parameters
lattice_size = 2000  # side length of the square-like lattice
initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))

T_steps = 1
B_steps = 1

Temperature = 2
B_field = -1
fixed_steps = 2

plot = False
images_directory = Find_Images_Directory()
save_directory = os.path.join(images_directory, f"lattice_{lattice_size}")

# Run the simulation
lattice_copy = np.copy(initial_lattice)
spin_lattice, magnetism = Ising_Model_Simulation(lattice_size, Temperature, fixed_steps, lattice_copy, B_field, plot)

Plot(initial_lattice, 'Initial Spin Configuration')
Plot_save(initial_lattice, f'Initial Spin Configuration, Lattice size: {lattice_size}', save_directory, f'initial_steps{fixed_steps}_temp{Temperature}_B{B_field}.png')
Plot(spin_lattice, 'Final Spin Configuration')
Plot_save(spin_lattice, f'Final Spin Configuration, Lattice size: {lattice_size}', save_directory, f'final_steps{fixed_steps}_temp{Temperature}_B{B_field}.png')


