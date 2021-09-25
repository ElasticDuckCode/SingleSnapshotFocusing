#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import scipy.linalg as linalg
import scipy.signal as sps

def ula_steering_vector(sensor_count: int, wavelength: float, doa: float) -> np.ndarray:
    # assuming sensor spacing, d = 1
    return np.exp(2j * np.pi / wavelength * np.arange(sensor_count) * np.sin(doa))


def ula_measurement_matrix(sensor_count: int, wavelength: float, doa_list: list) -> np.ndarray:
    matrix = np.zeros([sensor_count, len(doa_list)], dtype=complex)
    for i, doa in enumerate(doa_list):
        matrix[:, i] = ula_steering_vector(sensor_count, wavelength, doa)
    return matrix

def build_synthetic_measurements(measurements, manifold_matrix):
    try:
        sensor_count, grid_size = manifold_matrix.shape
        predicted_signal = cp.Variable(shape=grid_size)
        objective = cp.Minimize(cp.norm1(predicted_signal))
        constraint = [
                (manifold_matrix @ predicted_signal)[:len(measurements)] == measurements
        ]
        problem = cp.Problem(objective, constraint)
        problem.solve(solver='SCIPY')
        predicted_signal = predicted_signal.value
        syn_measurements = manifold_matrix @ predicted_signal
    except cp.error.SolverError:
        return np.zeros(sensor_count), predicted_signal
    return syn_measurements, predicted_signal



def fill_hankel_by_rank_minimization(hankel_measurements: np.ndarray, manifold_matrix: np.ndarray,
        max_iter: int = 1, gamma: float = 1e-1, err: float = 0) -> np.ndarray:

    try:
        sensor_count, grid_size = manifold_matrix.shape
        hankel_indx = np.nonzero(hankel_measurements)
        W = np.ones(grid_size)
        for i in range(max_iter):
            predicted_signal = cp.Variable(shape=grid_size)
            objective = cp.Minimize(cp.norm1(cp.multiply(W, predicted_signal)))
            hankel_matrix = manifold_matrix @ cp.diag(predicted_signal) @ manifold_matrix.T
            constraint = [
                #cp.norm2(hankel_matrix[hankel_indx] - hankel_measurements[hankel_indx]) <= err,
                hankel_matrix[hankel_indx] == hankel_measurements[hankel_indx]
            ]
            problem = cp.Problem(objective, constraint)
            #problem.solve(verbose=False, solver='SCIPY')
            problem.solve(verbose=False, solver='ECOS')
            predicted_signal = predicted_signal.value
            W = 1 / (np.abs(predicted_signal) + gamma)
            if i == 0:
                predicted_it0 = predicted_signal
            if i % 2 and gamma > 1e-5:
                gamma /= 10
        hankel_matrix = manifold_matrix @ np.diag(predicted_signal) @ manifold_matrix.T
    except cp.error.SolverError:
        return np.zeros_like(hankel_measurements), predicted_signal, predicted_signal

    return hankel_matrix, predicted_signal, predicted_it0


def ula_music(data: np.ndarray, source_count: int, wavelength: float, frequency_list: list) -> np.ndarray:
    sensor_count, _ = data.shape
    music_matrix = ula_measurement_matrix(sensor_count, wavelength, frequency_list)
    left_sing_vec, _, _ = linalg.svd(data, full_matrices=True)
    noise_subspace = left_sing_vec[:, source_count:]
    projection = linalg.norm(noise_subspace.conj().T @ music_matrix, axis=0)
    pseudospectrum = 1/projection
    pseudospectrum /= pseudospectrum.max()
    return pseudospectrum


def solve_mmv(measurements: np.ndarray, manifold_matrix: np.ndarray, err: float = 1e-10) -> np.ndarray:
    '''
    Solve multi-measurement optimization problem of the form

        min_X || X ||_2,1
        s.t   || Y - AX ||_F < err

        where Y = [y_1 ... y_t]^T
        and   X = [x_1 ... x_t]^T 

        for t signals sharing the same sparse support.

    '''
    sensor_count, grid_size = manifold_matrix.shape
    _, num_signals = measurements.shape
    predicted_signals = cp.Variable(shape=(grid_size, num_signals))
    objective = cp.Minimize(cp.mixed_norm(predicted_signals, 2, 1))
    constraints = [
        cp.norm(measurements - manifold_matrix @ predicted_signals, 'fro') <= err
        #measurements == manifold_matrix @ predicted_signals
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False, solver='ECOS')
    
    return predicted_signals.value.T


def solve_l1(measurement: np.ndarray, manifold_matrix: np.ndarray, err: float = 1e-10) -> np.ndarray:
    '''
    Solves constrained l1 optimization problem of the form

        min_x || x ||_1
        s.t.  || y - Ax ||_2 < err

        where the measurement model is 

            y = Ax
    '''
    try:
        sensor_count, grid_size = manifold_matrix.shape
        predicted_signal = cp.Variable(shape=(grid_size))
        objective = cp.Minimize(cp.norm(predicted_signal, 1))
        constraints = [
                #cp.norm(measurement - manifold_matrix @ predicted_signal, 2) <= err
                measurement == manifold_matrix @ predicted_signal
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False, solver='SCIPY')
    except cp.error.SolverError:
        return np.zeros(grid_size)
    return predicted_signal.value


def get_largest_k_peaks(signal: np.ndarray, k: int = 1):
    peak_ind, peak_info = sps.find_peaks(np.abs(signal), height=0)
    peak_height = peak_info['peak_heights']
    peak_sortind = peak_ind[peak_height.argsort()]
    pred_peaks = peak_sortind[-k:]
    return pred_peaks


def calculate_support_error(pred_signal, true_signal):
    true_peaks = np.nonzero(true_signal)[0]
    num_peaks = len(true_peaks)
    pred_peaks = get_largest_k_peaks(pred_signal, num_peaks)
    return np.mean(np.in1d(true_peaks, pred_peaks))

def calculate_nmse(pred_signal, true_signal):
    return linalg.norm(pred_signal - true_signal) / linalg.norm(true_signal)
