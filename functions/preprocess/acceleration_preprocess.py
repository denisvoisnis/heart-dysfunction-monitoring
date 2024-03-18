import sys
import pandas as pd
from scipy import signal
import os.path
from os import path
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt

def acceleration_preprocess(acceleration_axis, metadata):
    """ Preprocess acceleration signal.

    Parameters
    ----------
    acceleration_axis : np.array
        Acceleration signal of one axis.
    metadata : dict
        metadata of signal.

    Returns
    -------
    acceleration_axis_filt : np.array
        Filtered acceleration signal.
    """
    b, a = signal.butter(N=4, Wn=0.1, btype="highpass", fs=metadata['acc_fs'])
    acceleration_axis_filt = signal.filtfilt(b, a, acceleration_axis)

    N_sav_gol = int(metadata['acc_fs']/5)
    if (N_sav_gol % 2) == 0:
        N_sav_gol = N_sav_gol + 1
    # Savitzky-Golay filter
    acceleration_axis_filt = signal.savgol_filter(acceleration_axis_filt, N_sav_gol, 2)/1000
    
    return acceleration_axis_filt

def acceleration_magnitude(signals):
    """ Calculate magnitude of acceleration signal.

    Parameters
    ----------
    signals : np.array
        acceleration dataframe.

    Returns
    -------
    acc_mod : np.array
        Acceleration magnitude.
    """
    square_signals = [np.square(x) for x in signals]
    
    acc_mod_or = np.sqrt(np.sum(square_signals, axis=0))
    
    acc_mod = (np.array(acc_mod_or - np.min(acc_mod_or)))
    
    return acc_mod

def rotation_matrix_from_vectors(vec1, vec2):
    """ Create rotation matric.

    Parameters
    ----------
    vect1 : np.array
        reference vector.
    vect2 : np.array
        vector to rotate.
    Returns
    -------
    rotation_matrix: np.matrix
        Rotation matrix.
    """
    
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def rotate_acceleration_axis(acc):
    """ Rotate acceleration signal to constant direction.

    Parameters
    ----------
    acc : pd.DataFrame ['x','y','z']
        Acceleration dataframe.

    Returns
    -------
    acc_rotated : pd.DataFrame
        Retotated acceleration dataframe.
    """
    
    # Get rolling median filter of original acceleration axis directions
    med_acc_segment = acc.rolling(50*6, center = True, min_periods = 25).median()
    
    # the reference direction of acceleartion vector (from belt)
    vector_reference = [-990,   60,  150 ]
    vectors_original = acc.values
    vectors_compare = med_acc_segment.values
    
    # Create rotation matrices
    rotation_matrices = [rotation_matrix_from_vectors(vector_reference, v_n) for v_n in vectors_compare]
    
    # Rotate original acceleration signal
    r = R.from_matrix(rotation_matrices)
    vector_rotated = r.apply(vectors_original, inverse=True)
    
    # Create dataframe of rotated acceleration values
    acc_rotated = pd.DataFrame(vector_rotated, columns = ['x','y','z'])
    
    return acc_rotated

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y