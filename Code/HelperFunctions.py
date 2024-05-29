import os
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def Plot(lattice, title):
    size = lattice.shape[0]
    plt.figure(figsize=(size/5, size/5))
    plt.imshow(lattice, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

def Plot_save(lattice, title, save_directory, filename):
    size = lattice.shape[0]
    plt.figure(figsize=(size/5, size/5))
    plt.imshow(lattice, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, filename))
    plt.close()

def Find_Images_Directory():
    for root, dirs, files in os.walk('..'):
        if 'Images' in dirs:
            return os.path.join(root, 'Images')
    raise FileNotFoundError("Images directory not found")

@jit(nopython=True)
def Calculate_Energy(spin_lattice, B_field, lattice_size):
    energy = 0.0
    for i in range(lattice_size):
        for j in range(lattice_size):
            spin = spin_lattice[i, j]
            neighbor_sum = (
                  spin_lattice[(i + 1) % lattice_size, j]
                + spin_lattice[i, (j + 1) % lattice_size]
                + spin_lattice[(i - 1) % lattice_size, j]
                + spin_lattice[i, (j - 1) % lattice_size]
            )
            energy += -spin * neighbor_sum / 2 - B_field * spin
    return energy

@jit(nopython=True)
def Calculate_Magnetism(spin_lattice):  
    return np.sum(spin_lattice)

def Estimate_Largest_Cluster(spin_lattice):
    labeled_array_up, num_features_up = label(spin_lattice == 1)
    sizes_up = [np.sum(labeled_array_up == i) for i in range(1, num_features_up + 1)]
    
    labeled_array_down, num_features_down = label(spin_lattice == -1)
    sizes_down = [np.sum(labeled_array_down == i) for i in range(1, num_features_down + 1)]
    
    largest_cluster_size_up = max(sizes_up) if sizes_up else 0
    largest_cluster_size_down = max(sizes_down) if sizes_down else 0
    
    largest_cluster_size = max(largest_cluster_size_up, largest_cluster_size_down)
    
    return largest_cluster_size

def Create_Gif_From_Frames(frames, output_gif_path, fps=10, time_interval=1000):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    def update_frame(i):
        ax.clear()
        ax.axis('off')
        ax.imshow(frames[i], cmap='binary', interpolation='nearest')
        ax.set_title('Time Step: ' + str(i * time_interval))

    ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=time_interval/fps)
    ani.save(output_gif_path, writer='imagemagick')

def Plot_Largest_Cluster_Size(temperature_cluster_sizes, title, save_directory=None, filename=None):
    temperatures = list(temperature_cluster_sizes.keys())
    cluster_sizes = list(temperature_cluster_sizes.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, cluster_sizes, marker='o', linestyle='-')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Largest Cluster Size')
    plt.title(title)
    plt.grid(True)
    
    if save_directory and filename:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(os.path.join(save_directory, filename))
        plt.close()
    else:
        plt.show()


def Plot_Cluster_Size_vs_Temperature_Multiple_Lattices(results, title, save_directory=None, filename=None):
    plt.figure(figsize=(10, 6))
    for lattice_size, temperature_cluster_sizes in results.items():
        temperatures = list(temperature_cluster_sizes.keys())
        cluster_sizes = list(temperature_cluster_sizes.values())
        plt.plot(temperatures, cluster_sizes, marker='o', linestyle='-', label=f'Lattice size {lattice_size}')
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Largest Cluster Size')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_directory and filename:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(os.path.join(save_directory, filename))
        plt.close()
    else:
        plt.show()