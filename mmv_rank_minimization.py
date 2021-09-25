#!/usr/bin/env python3

import sys
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import ula_measurement_matrix, fill_hankel_by_rank_minimization, solve_mmv, solve_l1, calculate_nmse, calculate_support_error

def run_one_experiment(sparsity_level: int, sig: float, sensor_count: int,
        grid_size: int , wavelengths: list):
    signals = []                                                                                # Create two sparse signals which share the same DOA's, but have 
    signals.append(np.sign(np.random.randn(sparsity_level))) #np.ones(sparsity_level)           # different wavelengths
    signals.append(np.sign(np.random.randn(sparsity_level))) #np.ones(sparsity_level) 

    direction_grid = np.linspace(-np.pi/2, np.pi/2, grid_size)                                  # Assume that the DOA's lie on a grid, and randomly select a sparse amount
    direction_indx = np.random.choice(
            grid_size, 
            size=sparsity_level, 
            replace=False
    )

    #WARNING: This is assuming sparsity is 2
    sep = np.random.randint(1, grid_size//2 - 1)
    direction_indx = [grid_size//2 - sep, grid_size//2 + sep]
    directions = direction_grid[direction_indx]

    manifolds = []                                                                              # Create two different array manifold matricies for each of the 
    manifolds.append(ula_measurement_matrix(sensor_count, wavelengths[0], directions))          # wavelengths
    manifolds.append(ula_measurement_matrix(sensor_count, wavelengths[1], directions))

    noises = []                                                                                 # Each measurement will be contaminated with complex gaussian noise
    noises.append(sig * np.random.rand(sensor_count, 2).view(complex).ravel())
    noises.append(sig * np.random.rand(sensor_count, 2).view(complex).ravel())

    measurements = []                                                                           # Create measurements for the two wavelengths
    measurements.append(manifolds[0] @ signals[0] + noises[0])
    measurements.append(manifolds[1] @ signals[1] + noises[1])


    manifolds[0] = ula_measurement_matrix(sensor_count, wavelengths[0], direction_grid)           # Solve without doing any frequency extrapolatin/focusing
    manifolds[1] = ula_measurement_matrix(sensor_count, wavelengths[1], direction_grid)
    single_predictions = []
    single_predictions.append(solve_l1(
            measurements[0],
            manifolds[0],
            err=1.1*sig
        )
    )
    single_predictions.append(solve_l1(
            measurements[1],
            manifolds[1],
            err=1.1*sig
        )
    )

    hankels = []                                                                                # Now try to relate all measurements to same manifold by filling
    hankels.append(linalg.hankel(measurements[0]))                                              # in missing measurements of corresponding hankel matricies by
    measurements1_up = np.zeros(2*sensor_count - 1, dtype=complex)                              # performing rank minimization
    measurements1_up[::2] = measurements[1]
    hankels.append(linalg.hankel(measurements1_up[:sensor_count], measurements1_up[sensor_count-1:]))
    
    shared_manifold = ula_measurement_matrix(sensor_count, wavelengths[0], direction_grid)      # Explot full-grided shared_manifold to induce low-rank in the 
    filled_hankels = []                                                                         # hankel matricies. Then corresponding elements can be filled
    filled_hankels.append(                                                                      # by minimizing rank.
        fill_hankel_by_rank_minimization(hankels[0], manifolds[0], err=1.1*sig)
    )
    filled_hankels.append(
        fill_hankel_by_rank_minimization(hankels[1], manifolds[0], err=1.1*sig)
    )

    syn_measurements = np.zeros([2*sensor_count - 1, 2], dtype=complex)                             # Construct synthetic measurements from the hankel matrices,
    syn_measurements[:, 0] = np.concatenate([filled_hankels[0][:, 0], filled_hankels[0][-1, 1:]])   # and reconstruct the shared manifold for the larger measurement
    syn_measurements[:, 1] = np.concatenate([filled_hankels[1][:, 0], filled_hankels[1][-1, 1:]])   # count.
    shared_manifold = ula_measurement_matrix(2*sensor_count - 1, wavelengths[0], direction_grid)

    joint_predictions = solve_mmv(syn_measurements, shared_manifold, err=1.1*sig * np.sqrt(2))      # Now the the manifold is common to both measurements, solve using
                                                                                                    # multiple-measurement framework
    true_signals = []
    sig = np.zeros(grid_size)
    sig[direction_indx] = signals[0]
    true_signals.append(sig)
    sig = np.zeros(grid_size)
    sig[direction_indx] = signals[1]
    true_signals.append(sig)
                                                                                                 
    return np.asarray(single_predictions), np.asarray(joint_predictions), np.asarray(true_signals)


def monte_carlo_experiment() -> int:
    sensor_count = 16
    grid_size = 100
    sparsity_level = 2
    sigs = np.sqrt(np.logspace(-2.5, 2.5, 15) / 2) # complex noise
    wavelengths = [1, 1/2]
    n_monte_carlo = 100

    sig_nmse_smv = np.zeros(len(sigs))
    sig_nmse_mmv = np.zeros(len(sigs))
    sig_hit_smv = np.zeros(len(sigs))
    sig_hit_mmv = np.zeros(len(sigs))
    for i, sig in tqdm(enumerate(sigs)):
        avg_nmse_smv = 0
        avg_nmse_mmv = 0
        avg_hit_smv = 0
        avg_hit_mmv = 0
        for n in range(n_monte_carlo):
            single_pred, joint_pred, true_signal = run_one_experiment(
                    sparsity_level,
                    sig,
                    sensor_count,
                    grid_size,
                    wavelengths
            )
            avg_nmse_smv += 1/n_monte_carlo * calculate_nmse(single_pred, true_signal)
            avg_nmse_mmv += 1/n_monte_carlo * calculate_nmse(joint_pred, true_signal)
            avg_hit_smv  += 1/n_monte_carlo * 1/2 * (
                    calculate_support_error(single_pred[0], true_signal[0])
                    + calculate_support_error(single_pred[1], true_signal[1])
                    )
            avg_hit_mmv  += 1/n_monte_carlo * 1/2 * (
                    calculate_support_error(joint_pred[0], true_signal[0])
                    + calculate_support_error(joint_pred[1], true_signal[1])
                    )
        sig_nmse_smv[i] = avg_nmse_smv
        sig_nmse_mmv[i] = avg_nmse_mmv
        sig_hit_smv[i] = avg_hit_smv
        sig_hit_mmv[i] = avg_hit_mmv

    print(avg_nmse_smv, avg_nmse_mmv, avg_hit_smv, avg_hit_mmv)
    print(single_pred.shape, joint_pred.shape)

    db_sigs = 20 * np.log10(sigs)
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'font.sans-serif': ['Helvetica Neue', 'sans-serif']})

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(db_sigs, 10 * np.log10(sig_nmse_smv), color='b', marker='o', linewidth=2, markersize=8, markerfacecolor="None", label="SMV")
    ax.plot(db_sigs, 10 * np.log10(sig_nmse_mmv), color='r', marker='^', linewidth=2, markersize=8, markerfacecolor="None", label="MMV")
    ax.grid(color="#99aabb", linestyle=':')
    ax.set_facecolor("#e0f0ff")
    ax.set_xlabel("Noiser Power/dB")
    ax.set_xticks(np.arange(-25, 21, 5))
    ax.set_ylabel("NMSE")
    ax.set_yticks(np.arange(-15, 11, 5))
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/nmse.png")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(db_sigs, sig_hit_smv, color='b', marker='o', linewidth=2, markersize=8, markerfacecolor="None", label="SMV")
    ax.plot(db_sigs, sig_hit_mmv, color='r', marker='^', linewidth=2, markersize=8, markerfacecolor="None", label="SMV")
    ax.grid(color="#99aabb", linestyle=':')
    ax.set_facecolor("#e0f0ff")
    ax.set_xlabel("Noiser Power/dB")
    ax.set_xticks(np.arange(-25, 20+1, 5))
    ax.set_ylabel("Hit Rate")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/hit.png")
    return 0

def test() -> int:

    # Define program parameters
    sensor_count = 16
    grid_size = 100
    sparsity_level = 2
    noise_std = 1e-1 / np.sqrt(2) # sqrt(2) must compensate for complex numbers

    # signal will have two wavelengths
    wavelengths = [1, 1/2]

    # Create "sparse" signal
    signals = []
    signals.append(np.sign(np.random.randn(sparsity_level))) #np.ones(sparsity_level) 
    signals.append(np.sign(np.random.randn(sparsity_level))) #np.ones(sparsity_level) 

    # Assign DOA's
    direction_grid = np.linspace(-np.pi/2, np.pi/2, grid_size)
    direction_indx = np.random.choice(
            grid_size, 
            size=sparsity_level, 
            replace=False
    )
    directions = direction_grid[direction_indx]

    # Create array manifold
    manifold_matrix1 = ula_measurement_matrix(sensor_count, wavelengths[0], directions)
    manifold_matrix2 = ula_measurement_matrix(sensor_count, wavelengths[1], directions)

    # Measure signal w/ sensor array, introducing circular Gaussian noise
    noises = []
    noises.append(noise_std * np.random.rand(sensor_count, 2).view(complex).ravel())
    noises.append(noise_std * np.random.rand(sensor_count, 2).view(complex).ravel())
    measurements1 = manifold_matrix1 @ signals[0] + noises[0]
    measurements2 = manifold_matrix2 @ signals[1] + noises[1]

    # Fill measurement hankel matrices by viewing as "virtually coming from same array"
    shared_manifold = ula_measurement_matrix(sensor_count, wavelengths[0], direction_grid)

    measurements1_hankel = linalg.hankel(measurements1) # will fill rows
    hankel_matrix1 = fill_hankel_by_rank_minimization(measurements1_hankel, shared_manifold, err=1.1*noise_std)
    measurements2_upsample = np.zeros(2*sensor_count - 1, dtype=complex)
    measurements2_upsample[::2] = measurements2
    measurements2_hankel = linalg.hankel(measurements2_upsample[:sensor_count], measurements2_upsample[sensor_count-1:])
    hankel_matrix2 = fill_hankel_by_rank_minimization(measurements2_hankel, shared_manifold, err=1.1*noise_std)

    # Extract synthetic measurements
    syn_measurements = np.zeros([2*sensor_count - 1, 2], dtype=complex)
    syn_measurements[:, 0] = np.concatenate([hankel_matrix1[:, 0], hankel_matrix1[-1, 1:]])
    syn_measurements[:, 1] = np.concatenate([hankel_matrix2[:, 0], hankel_matrix2[-1, 1:]])
    shared_manifold = ula_measurement_matrix(2*sensor_count - 1, wavelengths[0], direction_grid)

    #errors1 = np.abs(syn_measurements[:, 0] - ula_measurement_matrix(2*sensor_count-1, wavelengths[0], directions) @ signals[0])
    #errors2 = np.abs(syn_measurements[:, 1] - ula_measurement_matrix(2*sensor_count-1, wavelengths[0], directions) @ signals[1])

    predicted_signals = solve_mmv(syn_measurements, shared_manifold, err=1.1*noise_std * np.sqrt(2))

    # Compare with solving individually w/o focusing
    manifold1 = ula_measurement_matrix(sensor_count, wavelengths[0], direction_grid)
    manifold2 = ula_measurement_matrix(sensor_count, wavelengths[1], direction_grid)
    single_prediction1 = solve_l1(measurements1, manifold1, err=2*noise_std * np.sqrt(sensor_count))
    single_prediction2 = solve_l1(measurements2, manifold2, err=2*noise_std * np.sqrt(sensor_count))
    
    # Plots
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'font.sans-serif': ['Helvetica Neue', 'sans-serif']})

    fig, ax = plt.subplots(2, 2, figsize=(7, 6))
    sig1 = np.zeros(grid_size)
    sig1[direction_indx] = signals[0]
    ax[0, 0].plot(np.degrees(direction_grid), sig1, 'b', linewidth=3, label='True')
    ax[0, 0].plot(np.degrees(direction_grid), predicted_signals[:, 0], 'r--', linewidth=3, label='MMV')
    ax[0, 1].plot(np.degrees(direction_grid), sig1, 'b', linewidth=3, label='True')
    ax[0, 1].plot(np.degrees(direction_grid), single_prediction1, 'g--', linewidth=3, label='SMV')

    sig2 = np.zeros(grid_size)
    sig2[direction_indx] = signals[1]
    ax[1, 0].plot(np.degrees(direction_grid), sig2, 'b', linewidth=3, label='True')
    ax[1, 0].plot(np.degrees(direction_grid), predicted_signals[:, 1], 'r--', linewidth=3, label='MMV')
    ax[1, 1].plot(np.degrees(direction_grid), sig2, 'b', linewidth=3, label='True')
    ax[1, 1].plot(np.degrees(direction_grid), single_prediction2, 'g--', linewidth=3, label='SMV')

    #plt.figure()
    #plt.plot(errors1)
    #plt.figure()
    #plt.plot(errors2)

    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(test())
    #sys.exit(monte_carlo_experiment())
