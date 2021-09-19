#!/usr/bin/env python3

# python modules
import os
import sys
from multiprocessing import Pool

# external modules
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal as sps
from tqdm import tqdm


def ula_array_manifold(sensor_count, normalized_frequency):
    return np.exp(1j * 2*np.pi * np.arange(sensor_count) * normalized_frequency)


def ula_measurement_matrix(sensor_count, frequency_list):
    matrix = np.zeros([sensor_count, len(frequency_list)], dtype=complex)
    for i, frequency in enumerate(frequency_list):
        matrix[:, i] = ula_array_manifold(sensor_count, frequency)
    return matrix


def ula_music(data, source_count, frequency_list):
    sensor_count, _ = data.shape
    music_matrix = ula_measurement_matrix(sensor_count, frequency_list)
    left_sing_vec, _, _ = linalg.svd(data, full_matrices=True)
    noise_subspace = left_sing_vec[:, source_count:]
    projection = linalg.norm(noise_subspace.conj().T @ music_matrix, axis=0)
    pseudospectrum = 1/projection
    pseudospectrum /= pseudospectrum.max()
    return pseudospectrum


def hankelize(measurements, num_measurements, rows):
    cols = num_measurements - rows

    cvx_matrix = np.zeros([rows, cols])
    cvx_matrix[:, 0] = measurements[:rows]
    cvx_matrix[-1, :] = measurements[rows:]
    for i in range(1, cols):
        for j in range(rows-1):
            cvx_matrix[j,i] = cvx_matrix[j+1,i-1]
    return


def find_optimum_hankel_nuclear_norm(measurements):
    m = len(measurements)
    H = cp.Variable(shape=(m, m), complex=True)
    hankel_measurements = linalg.hankel(measurements)
    indx = np.nonzero(hankel_measurements)
    objective = cp.Minimize(cp.normNuc(H))
    constraint = [
        H[indx] == hankel_measurements[indx],
        *[H[m-1,i] == H[m-1-j,i+j] for i in range(m) for j in range(m-i)]
    ]
    problem = cp.Problem(objective, constraint)
    problem.solve()

    #print("status:", problem.status)
    #print("optimal value", problem.value)
    return H.value


def find_optimum_hankel_reweighted(measurements, max_iter=10, gamma=1):
    m = len(measurements)
    hankel_measurements = linalg.hankel(measurements)
    indx = np.nonzero(hankel_measurements)
    W = np.eye(m)
    for i in tqdm(range(max_iter)):
        H = cp.Variable(shape=(m, m), complex=True)
        objective = cp.Minimize(cp.norm(W @ H, 'fro')**2)
        constraint = [
            H[indx] == hankel_measurements[indx],
            *[H[m-1,i] == H[m-1-j,i+j] for i in range(m) for j in range(m-i)]
        ]
        problem = cp.Problem(objective, constraint)
        problem.solve()
        H = H.value
        W = linalg.pinv(H.conj().T @ H + gamma*np.eye(H.shape[-1])) # p = 0
        W = linalg.sqrtm(W)
        #W = linalg.sqrtm(W) # uncomment for p = 1
        gamma /= 2
    #print("status:", problem.status)
    #print("optimal value", problem.value)
    return H


def find_optimum_hankel_reweighted_grid(measurements, measurement_matrix, max_iter=1, gamma=0.1, err=0):
    m, n = measurement_matrix.shape
    x = cp.Variable(shape=n)
    w = np.ones(n)
    hankel_measurements = linalg.hankel(measurements)
    indx = np.nonzero(hankel_measurements)
    for i in tqdm(range(max_iter)): # Assuming square hankel matrix
        objective = cp.Minimize(cp.norm1(cp.multiply(w, x)))
        #objective = cp.Minimize(cp.norm1(x))
        hankel_matrix = measurement_matrix @ cp.diag(x) @ measurement_matrix.T
        constraint = [
            cp.norm2(hankel_matrix[indx] - hankel_measurements[indx]) <= err,
            #hankel_matrix[indx] == hankel_measurements[indx],
            *[
                hankel_matrix[m-1,i] == hankel_matrix[m-1-j,i+j]
                for i in range(m) for j in range(m-i)
            ]
        ]
        problem = cp.Problem(objective, constraint)
        problem.solve()

        w = 1 / (np.abs(x.value) + gamma)
        gamma /= 10
    x = x.value
    hankel_matrix = measurement_matrix @ np.diag(x) @ measurement_matrix.T
    return hankel_matrix


def reduce_to_rank(matrix, new_rank):
    U, s, Vh = linalg.svd(matrix, full_matrices=True)
    s[new_rank:] = 0
    new_matrix = U @ linalg.diagsvd(s, U.shape[1], Vh.shape[0]) @ Vh
    return new_matrix


def run_experiment(measurement_matrix, sparsity_level, noise_std):
    #print("Running sig={}".format(noise_std))

    sensor_count, grid_size = measurement_matrix.shape

    # make sparse vector
    signal = np.zeros(grid_size)
    signal[:sparsity_level] = np.random.randn(sparsity_level)
    np.random.shuffle(signal)

    # make AWGN
    noise = noise_std * np.random.randn(sensor_count)

    # make measurements
    measurements = measurement_matrix @ signal + noise

    # infer frequency grid 
    fgrid = 1/grid_size * np.arange(grid_size)

    # complete hankel_matrix using rank-minimization
    new_hankel_matrix = find_optimum_hankel_nuclear_norm(measurements)
    music_spectrum = ula_music(new_hankel_matrix, sparsity_level, fgrid)

    # get signal support from peaks of music spectrum
    peak_index, peak_info = sps.find_peaks(music_spectrum, height=0)
    peak_heights = peak_info['peak_heights']
    peak_sortindex = peak_index[peak_heights.argsort()]
    predicted_peaks = peak_sortindex[-sparsity_level:]

    # solve least-squares to recover signal amplitudes
    tall_matrix = measurement_matrix[:, predicted_peaks]
    predicted_amplitudes = linalg.pinv(tall_matrix) @ measurements
    predicted_signal = np.zeros(grid_size)
    predicted_signal[predicted_peaks] = predicted_amplitudes.real

    nmse = linalg.norm(signal - predicted_signal) / linalg.norm(signal)

    return (predicted_signal, predicted_peaks, nmse);


def run_monte_carlo_experiments(num_experiments=10):
    sensor_count = 16
    sparsity_level = 2
    hankel_rows = sensor_count // 2
    grid_size = 100
    sigs = np.sqrt(np.logspace(-2.5, 2.5, 15))
    fgrid = 1/grid_size * np.arange(grid_size)
    measurement_matrix = ula_measurement_matrix(sensor_count, fgrid)

    nmse = np.zeros(len(sigs))

    # parallel processing
    for i in tqdm(range(num_experiments)):
        pool = Pool(os.cpu_count())
        data = pool.starmap(run_experiment, [
            (measurement_matrix, sparsity_level, sig) for sig in sigs
        ])

        nmse_losses = []
        for entries in data:
            nmse_losses.append(entries[-1])
        nmse_losses = np.asarray(nmse_losses)
        nmse += nmse_losses / num_experiments

    #plt.plot(sigs, 20*np.log10(nmse_losses))
    #plt.show()

    raise NotImplementedError


def main():
    '''
        Program to complete Hankel matrix for single-snapshot DOA/Frequency
        estimation problem.

        Measurement Model:
            y = Ax + n

            where A has exploitable Vandermonde structure, that would allow
            generation of extra measurements had the DOA/frequencies and
            amplitudes been known.

        In single snapshot methods, to be able to resolve M/2 sources, you 
        will have to at least have M measurements, as half the measurements
        will be sacrified in order to generate the rank needed for methods
        like Single-Snapshot MUSIC.

        Since the rank of the Hankel matrix can be assumed small due to 
        sparse support of X, missing measurements could be filled in by
        solving an optimization problem 

        (1)  min rank( H(y) )
             s.t. y_1, ..., y_M match correponding entries of H(y)

        where H(y) is the Hankelized version of the measurements.


        *----------------------------------------------------------------------*
        One way to solve (1) is to solve the surrogate nuclear norm problem

        (2)  min || H(y) ||_*
             s.t. y_1, ..., y_M match correponding entries of H(y)
        
        where || . ||_* denotes the nuclear norm.

        *----------------------------------------------------------------------*
        Another way to solve this (given the sources lie on a grid) is to 
        exploit the Vandermonde decomposition of a square Hankel matrix

        (3)  H(y) = A diag(x) A^T

        Note that we take the transpose, even when the array steering vectors
        are complex-valued.

        We know from (3) that A doesn't change, so x is the only thing
        controlling the rank of this matrix.

        Therefore, we can pose the normal compressed sensing problem

        (4)  min || x ||_0
             s.t. y_1, ..., y_M match corresponding entries of Adiag(x)A^T

        The objective above could be relaxed to the convex || . ||_1 norm, 
        or even use reweighted techniques like || Wx ||_1 or || Wx ||_2.

        for a set of iteratively updated weights W = diag(w)
    '''

    ''' Program Parameters '''
    sensor_count = 16
    sparsity_level = 2
    hankel_rows = sensor_count // 2
    grid_size = 100
    signal = np.random.randn(sparsity_level)
    print(f"{signal = }")
    noise_std = 0
    additive_noise = noise_std * np.random.randn(sensor_count)

    fgrid = 1/grid_size * np.arange(grid_size)
    freq_indx = np.random.choice(
        np.arange(grid_size),
        size=sparsity_level,
        replace=False
    )
    #freq_indx = [49, 51]
    frequencies = fgrid[freq_indx]
    measurement_matrix = ula_measurement_matrix(sensor_count, frequencies)
    measurements = measurement_matrix @ signal + additive_noise

    ''' Perform Single-Snapshot MUSIC'''
    hankel_matrix = linalg.hankel(
        measurements[:hankel_rows],
        measurements[hankel_rows-1:]
    )
    music_spectrum = ula_music(hankel_matrix, sparsity_level, fgrid)

    ''' Perform Nuclear Norm Minimization to find better Hankel Matrix '''
    #new_hankel_matrix = find_optimum_hankel_nuclear_norm(measurements)
    #new_music_spectrum = ula_music(new_hankel_matrix, sparsity_level, fgrid)

    ''' Perform Reweighted Trace Minimization to find better Hankel Matrix '''
    #new_hankel_matrix = find_optimum_hankel_reweighted(measurements, max_iter=30)
    #new_music_spectrum = ula_music(new_hankel_matrix, sparsity_level, fgrid)

    ''' Perform Reweighted Trace Minimization on Grid '''
    grid_matrix = ula_measurement_matrix(sensor_count, fgrid)
    new_hankel_matrix = find_optimum_hankel_reweighted_grid(
        measurements,
        grid_matrix,
        max_iter=5,
        err=1e-12
    )
    new_music_spectrum = ula_music(new_hankel_matrix, sparsity_level, fgrid)

    ''' Generate Synthetic Measurements We Created '''
    synthetic_measurements = np.concatenate([
        new_hankel_matrix[:,0],
        new_hankel_matrix[-1,1:]
    ])

    ''' Plot Results '''
    fig, ax = plt.subplots(3, figsize=(5, 7))
    ax[0].plot(music_spectrum, 'b', label="SS-MUSIC")
    ax[0].plot(new_music_spectrum, 'r', label="SS-MUSIC Synthetic")
    y_min, y_max = ax[0].get_ylim()
    ax[0].vlines(freq_indx, y_min-0.01, y_max+0.01, colors='k', linestyle='dashed')
    ax[0].legend()
    ax[0].set_title("SS-MUSIC Comparisons")

    new_sensor_count = len(synthetic_measurements)
    new_measurement_matrix = ula_measurement_matrix(new_sensor_count, frequencies)
    new_measurements = new_measurement_matrix @ signal
    errors = np.abs(new_measurements - synthetic_measurements)
    ax[1].semilogy(errors)
    ax[1].set_title("Measurement Errors")

    U, s, V = linalg.svd(new_hankel_matrix, full_matrices=True)
    ax[2].plot(s)
    ax[2].set_xticks(np.arange(len(s)))
    ax[2].set_title("Singular Values")
    plt.tight_layout()

    plt.figure()
    plt.subplot(121)
    h = np.abs(linalg.hankel(measurements))
    plt.imshow(h / h.max(), vmin=0, vmax=1)
    plt.title("Measurement Hankel Matrix")
    plt.subplot(122)
    nh = np.abs(new_hankel_matrix)
    plt.imshow(nh / nh.max(), vmin=0, vmax=1)
    plt.title("Filled Hankel Matrix")
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    #sys.exit(run_monte_carlo_experiments(2))
