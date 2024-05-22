import os
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def Plot(lattice, title):
    plt.figure(figsize=(7, 7))
    plt.imshow(lattice, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

def Plot_save(lattice, title, save_directory, filename):
    plt.figure(figsize=(7, 7))
    plt.imshow(lattice, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, filename))
    plt.close()

def Find_Images_Directory():
    for root, dirs, files in os.walk('.'):
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