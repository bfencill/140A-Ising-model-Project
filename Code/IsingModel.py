import numpy as np
from numba import jit
from HelperFunctions import Calculate_Energy, Calculate_Magnetism, Plot, Plot_save, Find_Images_Directory, Estimate_Largest_Cluster, Plot_Largest_Cluster_Size, Plot_Cluster_Size_vs_Temperature_Multiple_Lattices, Block_Spin_Transformation, Plot_Original_And_Transformed_Lattice, Create_Gif_From_Frames
from tqdm import tqdm
import os

np.random.seed(42)

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

def Ising_Model_Simulation(size, temperature, steps, spin_lattice, B_field, save_interval=0):
    frames = []
    magnetism = []
    for time_step in tqdm(range(steps), desc="Running Simulation"):
        MCMC_step(spin_lattice, temperature, B_field, size)
        magnetism.append(Calculate_Magnetism(spin_lattice))
        if (save_interval != 0) and (time_step % save_interval == 0):
            frames.append(np.copy(spin_lattice))
    return spin_lattice, magnetism, frames

def Ising_Model_Simulation_Block_Transformation(size, temperature, steps, spin_lattice, B_field, block_size, save_interval=0):
    frames = []
    magnetism = []
    transformed_size = size // block_size
    for time_step in tqdm(range(steps), desc="Running Simulation"):
        MCMC_step(spin_lattice, temperature, B_field, size)
        magnetism.append(Calculate_Magnetism(spin_lattice))
        if (save_interval != 0) and (time_step % save_interval == 0):
            # Block spin transformation
            transformed_lattice = Block_Spin_Transformation(spin_lattice, block_size)
            # Put the original and transformed lattice in the same frame, for visualization purposes
            # but keep them fairly separate in the image
            combined_lattice = np.zeros((size, 2*size))
            combined_lattice[:, :size] = spin_lattice
            combined_lattice[transformed_size:2*transformed_size, size + transformed_size: size + 2*transformed_size] = transformed_lattice
            frames.append(np.copy(combined_lattice))
    return spin_lattice, magnetism, frames

def Run_Simulation_For_Temperatures(lattice_size, temperatures, steps, B_field, discard_metastable=True):
    initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    plot = True
    images_directory = Find_Images_Directory()
    save_directory = os.path.join(images_directory, f"lattice_{lattice_size}")

    results = {}
    for temp in temperatures:
        lattice_copy = np.copy(initial_lattice)
        spin_lattice, magnetism, _ = Ising_Model_Simulation(lattice_size, temp, steps, lattice_copy, B_field)
        largest_cluster_size = Estimate_Largest_Cluster(spin_lattice)
        
        average_magnetism = np.mean(magnetism)
        saturation_magnetism = lattice_size * lattice_size
        average_magnetization_percentage = (average_magnetism / saturation_magnetism) * 100
        
        if discard_metastable:
            unique_values, counts = np.unique(spin_lattice, return_counts=True)
            if len(unique_values) == 2 and (counts[0] == 1 or counts[1] == 1):
                print(f"Temperature: {temp}, Metastable state detected and discarded.")
                continue
        
        results[temp] = {
            'final_lattice': spin_lattice,
            'magnetism': magnetism,
            'largest_cluster_size': largest_cluster_size,
            'average_magnetization_percentage': average_magnetization_percentage
        }

        Plot(spin_lattice, f'Final Spin Configuration at T={temp}')
        Plot_save(spin_lattice, f'Final Spin Configuration, Lattice size: {lattice_size}, T={temp}', save_directory, f'final_steps{steps}_temp{temp}_B{B_field}.png')
        
        print(f'Temperature: {temp}, Average Magnetization: {average_magnetization_percentage}%, Largest Cluster Size: {largest_cluster_size}')
        
    return results

def Run_Simulation_With_GIF(lattice_size, temperature, steps, B_field, save_interval, gif_output_path):
    initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    spin_lattice, magnetism, frames = Ising_Model_Simulation(lattice_size, temperature, steps, initial_lattice, B_field, save_interval)
    
    Create_Gif_From_Frames(frames, gif_output_path,time_interval=save_interval)

    Plot(spin_lattice, f'Final Spin Configuration at T={temperature}')
    images_directory = Find_Images_Directory()
    save_directory = os.path.join(images_directory, f"lattice_{lattice_size}_T{temperature}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    Plot_save(spin_lattice, f'Final Spin Configuration, Lattice size: {lattice_size}, T={temperature}', save_directory, f'final_steps{steps}_temp{temperature}_B{B_field}.png')

    return spin_lattice, magnetism

def Calculate_Cluster_Size_vs_Temperature(lattice_size, temperatures, steps, B_field, save_interval=0):
    initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    temperature_cluster_sizes = {}
    for temp in tqdm(temperatures, desc="Running Simulations for Various Temperatures"):
        spin_lattice, magnetism, _ = Ising_Model_Simulation(lattice_size, temp, steps, np.copy(initial_lattice), B_field, save_interval)
        largest_cluster_size = Estimate_Largest_Cluster(spin_lattice)
        temperature_cluster_sizes[temp] = largest_cluster_size
    return temperature_cluster_sizes

def Calculate_Cluster_Size_vs_Temperature_Multiple_Lattices(lattice_sizes, temperatures, steps, B_field, save_interval):
    results = {}
    for lattice_size in lattice_sizes:
        initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
        temperature_cluster_sizes = {}
        for temp in tqdm(temperatures, desc=f"Running Simulations for Lattice Size {lattice_size}"):
            spin_lattice, magnetism, _ = Ising_Model_Simulation(lattice_size, temp, steps, np.copy(initial_lattice), B_field, save_interval)
            largest_cluster_size = Estimate_Largest_Cluster(spin_lattice)
            temperature_cluster_sizes[temp] = largest_cluster_size
        results[lattice_size] = temperature_cluster_sizes
    return results

def Run_Simulation_With_Block_Spin_Transformation(lattice_size, temperature, steps, B_field, save_interval=0, gif_output_path='', block_size=3):
    initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    spin_lattice, magnetism, frames = Ising_Model_Simulation_Block_Transformation(lattice_size, temperature, steps, initial_lattice, B_field, block_size, save_interval)
    largest_cluster_size = Estimate_Largest_Cluster(spin_lattice)

    if gif_output_path:
        Create_Gif_From_Frames(frames, gif_output_path, two_lattices=True, time_interval=save_interval)
    
    transformed_lattice = Block_Spin_Transformation(spin_lattice, block_size)
    transformed_largest_cluster_size = Estimate_Largest_Cluster(transformed_lattice) * block_size * block_size
    
    Plot_Original_And_Transformed_Lattice(spin_lattice, transformed_lattice, f'Original and Transformed Lattice at T={temperature}')
    magnetism_transformed = Calculate_Magnetism(transformed_lattice)

    return spin_lattice, transformed_lattice, magnetism, magnetism_transformed, largest_cluster_size, transformed_largest_cluster_size

# Example usage
if __name__ == "__main__":
    lattice_size = 20
    temperatures = [2, 1.5, 1]
    steps_per_dipole = 100
    B_field = -1
    results = Run_Simulation_For_Temperatures(lattice_size, temperatures, steps_per_dipole, B_field)
