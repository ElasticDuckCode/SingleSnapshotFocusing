#!/usr/bin/env python3

import sys
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import ula_measurement_matrix, fill_hankel_by_rank_minimization, solve_mmv, solve_l1, calculate_nmse, calculate_support_error

def main() -> int:
    n_sensors = 16
    n_grid = 100
    n_sparity = 2
    noise_std = 0 / np.sqrt(2)
    wavelengths = [1, 1/2, 1/3]
    n_wavelengths = len(wavelengths)

    doa_grid = np.linspace(-np.pi/2, np.pi/2, n_grid)
    doa_indx = np.random.choice(n_grid, size=n_sparity, replace=False)
    doa = doa_grid[doa_indx]

    signals = np.zeros([n_wavelengths, n_grid])
    signals[0][doa_indx] = np.sign(np.random.randn(n_sparity))
    signals[1][doa_indx] = np.sign(np.random.randn(n_sparity))
    signals[2][doa_indx] = np.sign(np.random.randn(n_sparity))

    manifolds = np.zeros([n_wavelengths, n_sensors, n_grid], dtype=complex)
    manifolds[0] = ula_measurement_matrix(n_sensors, wavelengths[0], doa_grid)
    manifolds[1] = ula_measurement_matrix(n_sensors, wavelengths[1], doa_grid)
    manifolds[2] = ula_measurement_matrix(n_sensors, wavelengths[2], doa_grid)

    noises = noise_std * np.random.rand(n_wavelengths, n_sensors, 2).view(complex).squeeze()

    measurements = np.zeros([n_wavelengths, n_sensors], dtype=complex)
    measurements[0] = manifolds[0] @ signals[0] + noises[0]
    measurements[1] = manifolds[1] @ signals[1] + noises[1]
    measurements[2] = manifolds[2] @ signals[2] + noises[2]
    
    n_syn_sensors = n_wavelengths*n_sensors - n_wavelengths + 1
    hankels = np.zeros([n_wavelengths, n_syn_sensors//2 + 1, n_syn_sensors//2 + 1], dtype=complex)
    # need to relate each of the wavelengths
    m0 = np.zeros(n_syn_sensors, dtype=complex)
    m0[:n_sensors:1] = measurements[0]
    hankels[0] = linalg.hankel(m0[:n_syn_sensors//2 + 1])
    m1 = np.zeros(n_syn_sensors, dtype=complex)
    m1[:2*n_sensors:2] = measurements[1]
    hankels[1] = linalg.hankel(m1[:n_syn_sensors//2 + 1], m1[n_syn_sensors//2 - 1:])
    m2 = np.zeros(n_syn_sensors, dtype=complex)
    m2[:3*n_sensors:3] = measurements[2]
    hankels[2] = linalg.hankel(m2[:n_syn_sensors//2 + 1], m2[n_syn_sensors//2 - 1:])

    plt.imshow(hankels[2] > 0)
    plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())
