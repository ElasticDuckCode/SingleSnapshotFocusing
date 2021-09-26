#!/usr/bin/env python3

import sys
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import ula_measurement_matrix, fill_hankel_by_rank_minimization, solve_mmv, solve_l1, calculate_nmse, calculate_support_error

def pulaks_prediction(hankel, measurement, k):
    M = len(measurement)
    syn_meas = measurement
    Phi = hankel[:, :k]
    L0 = M//2 + 1
    L = L0
    PhiL = Phi[:L]
    for i in range(M+1):
        c = Phi[L0] @ linalg.pinv(PhiL) @ syn_meas[L-1:M]
        syn_meas = np.append(syn_meas, c)
        print(i, syn_meas.shape)
        L += 1
        M += 1
    return syn_meas

def other_main(plot=True):
    n_sensors = 15
    n_grid = 100
    n_sparity = 3
    noise_std = 0 / np.sqrt(2)
    wavelengths = [1, 1/2, 1/3]
    n_wavelengths = len(wavelengths)

    doa_grid = np.linspace(0, np.pi/4, n_grid)
    doa_indx = np.random.choice(n_grid, size=n_sparity, replace=False)
    #doa_indx = np.arange(50, 50+2*n_sparity, 2)
    #doa_indx = np.asarray([72, 73, 74])  #11, 12, 13 bad
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

    pred_doa0 = solve_l1(measurements[0], manifolds[0])
    plt.figure()
    plt.plot(doa_grid, signals[0])
    plt.plot(doa_grid, pred_doa0)

    MSE = 1/n_grid * np.linalg.norm(pred_doa0 - signals[0])

    return MSE


def main(plot=True) -> int:
    n_sensors = 15
    n_grid = 100
    n_sparity = 3
    noise_std = 0 / np.sqrt(2)
    wavelengths = [1, 1/2, 1/3]
    wavelengths = 1 / np.arange(1, 8)
    #wavelengths = [1, 1/2]
    n_wavelengths = len(wavelengths)

    doa_grid = np.linspace(0, np.pi/4, n_grid)
    doa_indx = np.random.choice(n_grid, size=n_sparity, replace=False)
    #doa_indx = np.arange(50, 50+2*n_sparity, 2)
    #doa_indx = np.asarray([72, 73, 74])  #11, 12, 13 bad
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
    #n_syn_sensors = 2*n_sensors + 1

    syn_measurements = np.zeros([n_wavelengths, n_syn_sensors], dtype=complex)
    hankels = np.zeros([n_wavelengths, n_syn_sensors//2 + 1, n_syn_sensors - n_syn_sensors//2], dtype=complex)

    syn_measurements[0][0:n_sensors] = measurements[0]
    syn_measurements[1][0:2*n_sensors:2] = measurements[1]
    syn_measurements[2][0:3*n_sensors:3] = measurements[2]

    hankels[0] = linalg.hankel(syn_measurements[0][:n_syn_sensors//2+1], syn_measurements[0][n_syn_sensors//2:])
    hankels[1] = linalg.hankel(syn_measurements[1][:n_syn_sensors//2+1], syn_measurements[1][n_syn_sensors//2:])
    hankels[2] = linalg.hankel(syn_measurements[2][:n_syn_sensors//2+1], syn_measurements[2][n_syn_sensors//2:])

    shared_manifold = ula_measurement_matrix(hankels.shape[1], wavelengths[0], doa_grid)
    filled_hankels = np.zeros_like(hankels)
    filled_hankels[0], predicted_doa_signal, predicted_it0 = fill_hankel_by_rank_minimization(hankels[0], shared_manifold)
    filled_hankels[1], predicted_doa_signal, predicted_it0 = fill_hankel_by_rank_minimization(hankels[1], shared_manifold)
    filled_hankels[2], predicted_doa_signal, predicted_it0 = fill_hankel_by_rank_minimization(hankels[2], shared_manifold)

    syn_measurement1 = np.zeros(n_syn_sensors, dtype=complex)
    syn_measurement1 = np.concatenate([filled_hankels[2][:, 0], filled_hankels[2][-1, 1:]])

    shared_manifold = ula_measurement_matrix(n_syn_sensors, wavelengths[0], doa_grid)
    true_measurement1 = shared_manifold @ signals[2]
    true_hankel1 = linalg.hankel(true_measurement1[:n_syn_sensors//2+1], true_measurement1[n_syn_sensors//2:])
    #does_it_match = manifolds[0] @ predicted_doa_signal

    #new_hank = linalg.hankel(syn_measurement1)
    #shared_manifold = ula_measurement_matrix(new_hank.shape[1], wavelengths[0], doa_grid)
    #fill_again, pdoas, pit0 = fill_hankel_by_rank_minimization(new_hank, shared_manifold)

    

    #plt.figure()
    #plt.imshow(np.abs(hankels[0]))
    #plt.figure()
    #plt.imshow(np.abs(filled_hankels[0]))
    #plt.figure()
    #plt.imshow(np.abs(new_hank))
    #plt.figure()
    #plt.imshow(np.abs(fill_again))
    #plt.show()

    #syn_measurement1 = np.concatenate([fill_again[:, 0], fill_again[-1, 1:]])
    #shared_manifold = ula_measurement_matrix(2*n_syn_sensors - 1, wavelengths[0], doa_grid)
    #true_measurement1 = shared_manifold @ signals[0]

    #MSE = 1/n_syn_sensors * linalg.norm(syn_measurement1 -true_measurement1)
    #if (MSE > 1e-4):
    #    plot = True

    doaplot = np.zeros_like(doa_grid)
    doaplot[doa_indx] = signals[0][doa_indx]
    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(131)
        img = np.abs(hankels[2])
        plt.imshow(img / img.max(), cmap='magma', vmin=0.0, vmax=1.0)
        plt.title('Missing Measurement Hankel')
        plt.subplot(132)
        img = np.abs(filled_hankels[2])
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
    return


if __name__ == "__main__":
    main(plot=True)

