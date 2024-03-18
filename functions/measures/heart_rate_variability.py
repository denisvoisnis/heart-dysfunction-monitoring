# imports
import pandas as pd
import numpy as np
import os
import sys
from scipy import signal

from sklearn.linear_model import LinearRegression

from scipy import interpolate, signal
from scipy.interpolate import CubicSpline

def frequency_domain(ppi, fs=4):
    """ Evaluate heart rate variability measures.
    
    Parameters
    ----------
    ppi : np.array
        P-P intervals.
    fs : int
        sampling rate of P-P intervals.
        
    Returns
    -------
    results : dict
        Dictionary of HRC results.
    fxx : np.array
        Frequency axis of power spectrum 
    pxx : np.array
        Power spectrum of axis
    """
    
    # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x=ppi, fs=fs, nperseg = len(ppi))
    
    # Segement found frequencies in the bands 
    # - Low Frequency (LF): 0.04-0.15Hz 
    # - High Frequency (HF): 0.15-0.4Hz
    
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)
    
    # calculate power in each band by integrating the spectral density 
    lf = np.trapz(pxx[cond_lf], fxx[cond_lf])
    hf = np.trapz(pxx[cond_hf], fxx[cond_hf])
    
    results = {}
    results['Power_LF'] = lf # (ms2)
    results['Power_HF'] = hf # (ms2)  
    
    results['LF/HF'] = (lf/hf)
    
    return results, fxx, pxx

def correct_ectopic_beats(pp):
    """ Correction of ecxtopic and abnormal beats in the heart rate series.
    
    Parameters
    ----------
    pp : np.array
        pp intervals of ppg signal.
    
    Returns
    -------
    pp_good_corrected : np.array
        Corrected pp intervals of ppg signal. 
    """
    
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from utils import ranges, blockPrint, enablePrint
    from remove_ectopic_beats import detect_ectopics
    
    blockPrint()
    bad_beats_recovery = detect_ectopics(pp)
    enablePrint()
    pp_good_removed = np.delete(pp, bad_beats_recovery)
    pp_good_corrected = pp.copy()
    std_pp = np.std(pp_good_removed)
    # Replace to an median value
    rand_array = (np.random.rand(len(bad_beats_recovery))-0.5)*(2*std_pp) + np.median(pp)
    pp_good_corrected[bad_beats_recovery] = rand_array
    
    return pp_good_corrected


def estimate_heart_rate_variability(pp_arr, pp_time_arr):
    """ Correction of ecxtopic and abnormal beats in the heart rate series.
    
    Parameters
    ----------
    pp_arr : np.array
        pp intervals of ppg signal.
    pp_time_arr : np.array
        indexes of pp intervals of ppg signal.
        
    Returns
    -------
    result : np.array
        Corrected pp intervals of ppg signal. 
    """

    PP_baseline = np.round(np.mean(pp_arr),1)
    
    ## Calculate heart rate variability
    try:
        spl = CubicSpline(pp_time_arr, pp_arr)
        fs_pp = 4.0
        steps = 1 / fs_pp
        
        xx = np.arange(1, np.max(pp_time_arr), steps)
        pp_interpolated = spl(xx[1:])
        
        SDNN = np.round(np.std(pp_arr),2)    
        
        results, fxx, pxx = frequency_domain(pp_arr, fs_pp)
        lf_hf = np.round(results['LF/HF'],6)
        
    except: ## Unable to interpolate and calculate HRV
        SDNN = np.nan
        lf_hf = np.nan
    
    results = {'SDNN': SDNN, 'lf_hf': lf_hf, 'HR_baseline': np.round(60/(PP_baseline/1000),2)}
    
    return results
    


def correct_linearity(pp_arr, pp_time_arr):
    """ Correction of of linearity of PP or RR intervals.
    
    Parameters
    ----------
    pp_arr : np.array
        pp intervals of ppg signal.
    pp_time_arr : np.array
        indexes of pp intervals of ppg signal.
        
    Returns
    -------
    pp_arr_cor : np.array
        corrected pp intervals of ppg signal.
    pp_time_arr_cor : np.array
        corrected indexes of pp intervals of ppg signal.
    """
    
    # find unlinear points
    diffs_time = np.diff(pp_time_arr)
    
    # find that are close to 0 or negative
    inds_nonlinear = np.where(diffs_time < 0.01)[0]
    
    
    
    pp_arr_cor = np.delete(pp_arr, inds_nonlinear)
    pp_time_arr_cor = np.delete(pp_time_arr, inds_nonlinear)
    
    return pp_arr_cor, pp_time_arr_cor



def evaluate_heart_rate_variability(peakIndArr, sqi, metadata):
    """ Evaluate heart rate variability measures.
    
    Parameters
    ----------
    acc : df
        acceleration dataframe.
    ppg : df
        photoplethymogram dataframe.
    walking_bouts : df
        dataframe of rest intervals information
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    results : dict
        Dictionary of heart rate variability results. 
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from utils import ranges, blockPrint, enablePrint
    from remove_ectopic_beats import detect_ectopics
    
    ## Extract good quality indices
    sqi_arr = np.array(sqi)[peakIndArr]
    sqi_arr_shift = np.roll(sqi_arr, -3)
    ind_arr = np.array(peakIndArr)[np.array(sqi_arr_shift).astype(bool)]
    
    pp_raw = np.diff(ind_arr)
    sqi_good_inds = np.where(np.array(sqi) == 1)[0]
    inds_good_ranges = ranges(sqi_good_inds)
    peakIndArr = np.array(peakIndArr)
    
    pp_good = np.array([])
    pp_ind_good = np.array([])
    for rn in inds_good_ranges:
        peakIndArr_sqi_ind = np.where((peakIndArr>rn[0]) & (peakIndArr<rn[1]))
        if len(peakIndArr_sqi_ind[0])> 1:
            with np.errstate(divide='ignore'):
                pp_add = 60/(np.diff(peakIndArr[peakIndArr_sqi_ind[0]])/metadata['ppg_resample_fs'])
            pp_good = np.concatenate((pp_good, pp_add))
            pp_ind_add = peakIndArr[peakIndArr_sqi_ind[0]][0:-1]/metadata['ppg_resample_fs']
            pp_ind_good = np.concatenate((pp_ind_good, pp_ind_add))
    
    ## Remove ectopic and false beats of HR recovery series
    pp_good_corrected = correct_ectopic_beats(pp_good)
    
    # convert to ms
    pp_good_corrected_ms = (60/(pp_good_corrected))*1000
    
    pp_good_corrected_ms, pp_ind_good = correct_linearity(pp_good_corrected_ms, pp_ind_good)
    
    results = estimate_heart_rate_variability(pp_good_corrected_ms, pp_ind_good)
    
    return results
    