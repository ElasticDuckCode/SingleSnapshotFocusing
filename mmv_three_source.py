#!/usr/bin/env python3

import sys
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import ula_measurement_matrix, fill_hankel_by_rank_minimization, solve_mmv, solve_l1, calculate_nmse, calculate_support_error

def main() -> int:
    n_sensors = 15
    n_grid = 100
    n_sparity = 5
    noise_std = 0 / np.sqrt(2)
    wavelengths = [1, 1/2, 1/3]
    n_wavelengths = len(wavelengths)

    doa_grid = np.linspace(-np.pi/2, np.pi/2, n_grid)
    doa_indx = np.random.choice(n_grid, size=n_sparity, replace=False)
    doa = doa_grid[doa_indx]

    signals = np.zeros([n_wavelengths, n_grid])
    signals[0][doa_indx] = np.random.randn(n_sparity)
    signals[1][doa_indx] = np.random.randn(n_sparity)
    signals[2][doa_indx] = np.random.randn(n_sparity)

    manifolds = np.zeros([n_wavelengths, n_sensors, n_grid], dtype=complex)
    manifolds[0] = ula_measurement_matrix(n_sensors, wavelengths[0], doa_grid)
    manifolds[1] = ula_measurement_matrix(n_sensors, wavelengths[1], doa_grid)
    manifolds[2] = ula_measurement_matrix(n_sensors, wavelengths[2], doa_grid)

    noises = noise_std * np.random.rand(n_wavelengths, n_sensors, 2).view(complex).squeeze()

    measurements = np.zeros([n_wavelengths, n_sensors], dtype=complex)
    measurements[0] = manifolds[0] @ signals[0] + noises[0]
    measurements[1] = manifolds[1] @ signals[1] + noises[1]
    measurements[2] = manifolds[2] @ signals[2] + noises[2]
    print(measurements.shape)
    
    n_syn_sensors = n_wavelengths*n_sensors - n_wavelengths + 1
    syn_measurements = np.zeros([n_wavelengths, n_syn_sensors], dtype=complex)
    hankels = np.zeros([n_wavelengths, n_syn_sensors//2 + 1, n_syn_sensors - n_syn_sensors//2], dtype=complex)

    syn_measurements[0][0:n_sensors] = measurements[0]
    syn_measurements[1][0:2*n_sensors:2] = measurements[1]
    syn_measurements[2][0:3*n_sensors:3] = measurements[2]

    hankels[0] = linalg.hankel(syn_measurements[0][:n_syn_sensors//2+1], syn_measurements[0][n_syn_sensors//2:])
    hankels[1] = linalg.hankel(syn_measurements[1][:n_syn_sensors//2+1], syn_measurements[1][n_syn_sensors//2:])
    hankels[2] = linalg.hankel(syn_measurements[2][:n_syn_sensors//2+1], syn_measurements[2][n_syn_sensors//2:])
    print(hankels.shape)
    print(manifolds[0].shape)

    shared_manifold = ula_measurement_matrix(hankels.shape[1], wavelengths[0], doa_grid)
    filled_hankels = np.zeros_like(hankels)
    filled_hankels[0] = fill_hankel_by_rank_minimization(hankels[0], shared_manifold)

    syn_measurement1 = np.zeros(n_syn_sensors, dtype=complex)
    syn_measurement1 = np.concatenate([filled_hankels[0][:, 0], filled_hankels[0][-1, 1:]])

    shared_manifold = ula_measurement_matrix(n_syn_sensors, wavelengths[0], doa_grid)
    true_measurement1 = shared_manifold @ signals[0]
    true_hankel1 = linalg.hankel(true_measurement1[:n_syn_sensors//2+1], true_measurement1[n_syn_sensors//2:])

    print("MSE: {}".format(1/n_syn_sensors * linalg.norm(syn_measurement1 -true_measurement1)))

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    img = np.abs(hankels[0])
    plt.imshow(img / img.max(), cmap='magma', vmin=0.0, vmax=1.0)
    plt.title('Missing Measurement Hankel')
    plt.subplot(132)
    img = np.abs(filled_hankels[0])
    plt.imshow(img / img.max(), cmap='magma', vmin=0.0, vmax=1.0)
    plt.title('Filled Hankel')
    plt.subplot(133)
    img = np.abs(true_hankel1)
    plt.imshow(img / img.max() , cmap='magma', vmin=0.0, vmax=1.0)
    plt.title('Ground Truth Hankel')
    plt.tight_layout()

    plt.figure()
    plt.plot(np.abs(true_measurement1 - syn_measurement1), linewidth=2)
    plt.xlabel('Sensor Index')
    plt.ylabel('Absolute Error')
    plt.title('Predicted VS True Measurement Error')
    plt.grid()
    plt.tight_layout()

    plt.figure()
    U, s, V = linalg.svd(filled_hankels[0])
    plt.plot(s)
    plt.show()


    return 0

if __name__ == "__main__":
    sys.exit(main())
