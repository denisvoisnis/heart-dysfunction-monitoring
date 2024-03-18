import sys
import pandas as pd
from scipy import signal
import os.path
from os import path
import numpy as np

def step_detection_TAF(acc_mod, fs, max_val = 150, min_val = 0):
    """Steps detection in acceleration signal.

    Parameters
    ----------
    acc_mod : np.array
        acceleration magnitude signal.
    fs : int
        Sampling rate of acceleartion signal.
    max_val : int
        Maximal value threshold for step detection.
    min_val : int
        Minimal value threshold for step detection.

    Returns
    -------
    steps_arr : np.array
        Indexes of detected steps.
    time_array : np.array
        Array indicating time of steps_arr in seconds.
    steps_min : np.array
        Steps per time interval array.
    num_arr : np.array
        Indicates physical activity intensity (0 - no physical acivity, 1 - light, 1 - medium, 3 - heavy).
    """
    ##### 

    width = int(round(fs/8))
    lim_step = int(round(fs/4))
    inds = [0,1]
    low = 0
    if acc_mod [0] > max_val:
        low = 1
    
    steps = [0]
    steps_amp = []
    
    for i_n in range(len(acc_mod)):
        if low:
            if acc_mod[i_n] < max_val:
                low = 0
                if (inds[-1]-inds[-2]) > width:
                    ind_step = ((inds[-1]-inds[-2])/2)+inds[-2]
                    if len(steps) > 1:
                        if (ind_step - steps[-1]) >lim_step:
                            steps.append(ind_step)
                            steps_amp.append(np.max(acc_mod[inds[-2]:inds[-1]])) 
                    else:
                        steps.append(ind_step)
                        steps_amp.append(np.max(acc_mod[inds[-2]:inds[-1]])) 
        else:
            if acc_mod[i_n] > max_val:
                inds.append(i_n)
                low = 1
    
    steps_arr = pd.Series(np.array(steps))
    sec = 60
    prop = 60/sec
    part_dur = sec*fs
    n_parts = int(len(acc_mod)/part_dur)
    
    num = []
    time = []
    
    for i in range(0,n_parts):
        index_st = i*part_dur
        index_en = (i+1)*part_dur
        time.append(index_st/fs)
        num.append(np.sum(steps_arr.between(index_st, index_en)))
        
    # Estimate physical intensities
    steps_min = np.array(num)*prop
    num_arr = np.array(num)
    num_arr[num_arr<(60/prop)] = 0
    num_arr[(num_arr>(59/prop)) & (num_arr<(100/prop))] = 1
    num_arr[(num_arr>(99/prop)) & (num_arr<(130/prop))] = 2
    num_arr[num_arr>(129/prop)] = 3
    
    time_array = np.array(time)
    return steps_arr, time_array, steps_min, num_arr


def calc_mad(acc, fs):
    """Calculate mean amplitude deviation.

    Parameters
    ----------
    acc : np.array
        Acceleration magnitude signal.
    fs : int
        Sampling rate of acceleartion signal.

    Returns
    -------
    mads : np.array
        Mean amplitude deviation every second.
    mean_mads : float
        Average of mean amplitude deviation.
    """
    acc = [x/980.665 for x in acc]
    mads = []
    for i in range(int(np.floor(len(acc)/fs))):
        sec = acc[fs*i:fs*i+fs]
        mad1 = [np.abs(x-np.mean(sec)) for x in sec]
        mad = np.sum(mad1)/len(sec)
        mads.append(mad)
        mean_mad = np.mean(mads)
    return mads, mean_mad

