# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import signal
import sys


def get_best_quality_ppg(ppg_extracted, metadata):
    """ Get ppg with the best signal quality index (SQI).

    Parameters
    ----------
    ppg_extracted : df
        photoplethysmogram dataframe segment.
    metadata : dict
        metadata of signal.

    Returns
    -------
    sgi_n : int
        PPG signal that have best quality according to SQI index.
    sqis : list
        List containing good quality PPG. 
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/detectors')
    from AF_PPG_detector import AF_PPG_detector
    
    if metadata['ppg_n'] > 1:
        ppg_signals = [ppg_extracted['ppg0'], ppg_extracted['ppg1'], ppg_extracted['ppg2']]
        sqis = []
        for n_sig in ppg_signals:
            outPPG, peakValArr, peakIndArr, sqi, det_pattern, out_pattern, corr_pattern = AF_PPG_detector(np.array(n_sig), metadata['ppg_resample_fs'])
            sqis.append(np.round(np.sum(sqi)/len(sqi),3))
        sig_to_analyse = np.argmax(sqis)
    else:
        sqis = []
        outPPG, peakValArr, peakIndArr, sqi, det_pattern, out_pattern, corr_pattern = AF_PPG_detector(ppg_extracted['ppg0'], metadata['ppg_resample_fs'])
        sqis.append(np.round(np.sum(sqi)/len(sqi),3))
        sig_to_analyse = 0
    
    return sig_to_analyse, sqis


def rest_segments_detection_rr(acc, rr, metadata):
    """ Analyse signal - find segments in rr and acc signals.

    Parameters
    ----------
    acc : df
        acceleration dataframe.
    rr : df
        rr dataframe.
    metadata : dict
        metadata of signal.

    Returns
    -------
    df_walking_bouts_updated : df
        DataFrame containing the informationm of physical activity segemnts.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/functions/preprocess')
    sys.path.insert(1,work_dir + '/functions/measures')
    sys.path.insert(1,work_dir + '/functions/detectors')
    from PA_algorithms import step_detection_TAF, calc_mad
    from AF_PPG_detector import AF_PPG_detector
    from acceleration_preprocess import acceleration_preprocess, acceleration_magnitude, rotate_acceleration_axis, butter_lowpass_filter
    from heart_rate_variability import evaluate_heart_rate_variability, estimate_heart_rate_variability
    
    ## update constants
    baseline_rest_duration = metadata['baseline_rest_duration'] + 1 # minutes
    max_steps_during_rest = metadata['max_steps_during_rest'] # steps
    
    # Rotate 
    acc_original_axis = acc[['x','y','z']]
    rotated_acc = rotate_acceleration_axis(acc_original_axis)
    
    position_configuration = 0
    if metadata['position'] == 'wrist':
        position_configuration = 1
    if metadata['position'] == 'chest':
        position_configuration = 2
    else:
        position_configuration = 0
        
    if position_configuration == 0 or position_configuration == 1:
        acc_mod_or = np.sqrt(np.sum(np.square(rotated_acc[['x']]), axis=1))
        acc_mod = np.array(acc_mod_or - np.mean(acc_mod_or))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, 50, 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 200, min_val = 150)     # fs_oh1, max_val = 100, min_val = 0
    
    if position_configuration == 2:
        # acceleration preprocess
        Acc_m = np.array(acc['y'])
        Acc_a = np.array(acc['x'])
        Acc_v = np.array(acc['z'])
            
        # Butterworth high-pass filter
        b, a = signal.butter(N=4, Wn=0.3, btype="highpass", fs = metadata['acc_fs'])
        Acc_m_filt = signal.filtfilt(b, a, Acc_m)
        Acc_a_filt = signal.filtfilt(b, a, Acc_a)
        Acc_v_filt = signal.filtfilt(b, a, Acc_v)
        
        acc_mod_or = acceleration_magnitude([Acc_v_filt, Acc_m_filt, Acc_a_filt])
        acc_mod = (np.array(acc_mod_or - np.mean(acc_mod_or)))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, 50, 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 200, min_val = 150)   # , max_val = 150, min_val = 0
    
    if len(steps_min) > 5:
        # Estimate physical activity measures
        rest_minutes = steps_min.copy()
        rest_minutes[rest_minutes<max_steps_during_rest] = 1
        rest_minutes[rest_minutes>max_steps_during_rest] = 0
        
        rest_minutes_arr = np.array(rest_minutes)
        
        T_durations = np.array([[time_array[i+2], time_array[i+baseline_rest_duration+1]] for i in range(len(time_array)-(baseline_rest_duration+1))])
        T_minutes = np.array([np.array(rest_minutes_arr[i:i+baseline_rest_duration]) for i in range(len(rest_minutes_arr)-(baseline_rest_duration+1))])
        if len(T_minutes) > 0:
            inds_rest = np.where(np.sum(T_minutes, axis = 1)==baseline_rest_duration)[0]
        else:
            inds_rest = np.array([])
    else:
        inds_rest = np.array([])
        
    if len(inds_rest) > 0: # If there are any resting intervals 
        ## determine optimal baseline rest interval based on MAD
        mads_inds = []
        for inds_test in inds_rest:
            inds_to_test = T_durations[inds_test]
            acc_to_test = acc_mod_filt[int(inds_to_test[0]*metadata['acc_fs']):int(inds_to_test[1]*metadata['acc_fs'])]
            mads_inds.append(calc_mad(acc_to_test, metadata['acc_fs'])[1])
        ind_min_activity_rest_phase = np.argmin(mads_inds)
        
        rest_duration_to_analyse = T_durations[inds_rest[ind_min_activity_rest_phase]]
        
        ## Evaluate HRV indicators
        
        ## extract baseline HR rest segment
        rr_baseline_rest = rr.loc[(rr['timestamp']>int(rest_duration_to_analyse[0])) & (rr['timestamp']<int(rest_duration_to_analyse[1]))]
        
        
        HRV_results = estimate_heart_rate_variability(rr_baseline_rest['rrms'], rr_baseline_rest['rrms'])
        rest_SQI = np.nan
        
    else: # set results to nan
        rest_SQI = np.nan
        HRV_results = {'SDNN': np.nan, 'lf_hf': np.nan, 'HR_baseline': np.nan}
        
    Rest_segment_analysis_results = {}
    Rest_segment_analysis_results.update({'rest_SQI': rest_SQI})
    Rest_segment_analysis_results.update(HRV_results)
    
    return Rest_segment_analysis_results





def rest_segments_detection(acc, ppg, metadata):
    """ Analyse signal - find segments.

    Parameters
    ----------
    acc : df
        acceleration dataframe.
    ppg : df
        photoplethysmogram dataframe.
    metadata : dict
        metadata of signal.

    Returns
    -------
    df_walking_bouts_updated : df
        DataFrame containing the informationm of physical activity segemnts.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/functions/preprocess')
    sys.path.insert(1,work_dir + '/functions/measures')
    sys.path.insert(1,work_dir + '/functions/detectors')
    from PA_algorithms import step_detection_TAF, calc_mad
    from AF_PPG_detector import AF_PPG_detector
    from acceleration_preprocess import acceleration_preprocess, acceleration_magnitude, rotate_acceleration_axis, butter_lowpass_filter
    from heart_rate_variability import evaluate_heart_rate_variability
    
    ## update constants
    baseline_rest_duration = metadata['baseline_rest_duration'] + 1 # minutes
    max_steps_during_rest = metadata['max_steps_during_rest'] # steps
    
    # Rotate 
    acc_original_axis = acc[['x','y','z']]
    rotated_acc = rotate_acceleration_axis(acc_original_axis)
    
    position_configuration = 0
    if metadata['position'] == 'wrist':
        position_configuration = 1
    if metadata['position'] == 'chest':
        position_configuration = 2
    else:
        position_configuration = 0
        
    if position_configuration == 0 or position_configuration == 1:
        acc_mod_or = np.sqrt(np.sum(np.square(rotated_acc[['x']]), axis=1))
        acc_mod = np.array(acc_mod_or - np.mean(acc_mod_or))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, 50, 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 200, min_val = 150)     # fs_oh1, max_val = 100, min_val = 0
    
    if position_configuration == 2:
        # acceleration preprocess
        Acc_m = np.array(acc['y'])
        Acc_a = np.array(acc['x'])
        Acc_v = np.array(acc['z'])
            
        # Butterworth high-pass filter
        b, a = signal.butter(N=4, Wn=0.3, btype="highpass", fs = metadata['acc_fs'])
        Acc_m_filt = signal.filtfilt(b, a, Acc_m)
        Acc_a_filt = signal.filtfilt(b, a, Acc_a)
        Acc_v_filt = signal.filtfilt(b, a, Acc_v)
        
        acc_mod_or = acceleration_magnitude([Acc_v_filt, Acc_m_filt, Acc_a_filt])
        acc_mod = (np.array(acc_mod_or - np.mean(acc_mod_or)))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, 50, 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 200, min_val = 150)   # , max_val = 150, min_val = 0
    
    if len(steps_min) > 5:
        # Estimate physical activity measures
        rest_minutes = steps_min.copy()
        rest_minutes[rest_minutes<max_steps_during_rest] = 1
        rest_minutes[rest_minutes>max_steps_during_rest] = 0
        
        rest_minutes_arr = np.array(rest_minutes)
        
        T_durations = np.array([[time_array[i+2], time_array[i+baseline_rest_duration+1]] for i in range(len(time_array)-(baseline_rest_duration+1))])
        T_minutes = np.array([np.array(rest_minutes_arr[i:i+baseline_rest_duration]) for i in range(len(rest_minutes_arr)-(baseline_rest_duration+1))])
        if len(T_minutes) > 0:
            inds_rest = np.where(np.sum(T_minutes, axis = 1)==baseline_rest_duration)[0]
        else:
            inds_rest = np.array([])
    else:
        inds_rest = np.array([])
        
    if len(inds_rest) > 0: # If there are any resting intervals 
        ## determine optimal baseline rest interval based on MAD
        mads_inds = []
        for inds_test in inds_rest:
            inds_to_test = T_durations[inds_test]
            acc_to_test = acc_mod_filt[int(inds_to_test[0]*metadata['acc_fs']):int(inds_to_test[1]*metadata['acc_fs'])]
            mads_inds.append(calc_mad(acc_to_test, metadata['acc_fs'])[1])
        ind_min_activity_rest_phase = np.argmin(mads_inds)
        
        rest_duration_to_analyse = T_durations[inds_rest[ind_min_activity_rest_phase]]
        
        ## extract baseline HR rest segment
        ppg_baseline_rest = ppg[int(rest_duration_to_analyse[0]*metadata['ppg_fs']):int(rest_duration_to_analyse[1]*metadata['ppg_fs'])]
        
        
        ## resample to 100 hz
        keys_ppg = [x for x in ppg_baseline_rest.keys() if 'ppg' in x]
        ppg_resample = pd.DataFrame([])
        t_duration = int(round(ppg_baseline_rest['timestamp'].iloc[-1]-ppg_baseline_rest['timestamp'].iloc[0]))
        for x_key in keys_ppg:
            ppg_to_resample = ppg_baseline_rest[x_key]
            
            num = int(metadata['ppg_resample_fs']*(t_duration))
            ppg_resampled = signal.resample(np.array(ppg_to_resample), num, domain='time')
            ppg_resample[x_key] = ppg_resampled
        t_ppg_resample = np.linspace(0,t_duration, len(ppg_resample))
        ppg_resample['timestamp'] = t_ppg_resample
        
        
        ## evaluate quality of ppg available signals
        sig_to_analyse, sqis = get_best_quality_ppg(ppg_resample, metadata)
        rest_SQI = sqis[sig_to_analyse]
        if metadata['ppg_n'] > 1:
            ppg_signals = [ppg_resample['ppg0'], ppg_resample['ppg1'], ppg_resample['ppg2']]
        else:
            ppg_signals = [ppg_resample['ppg0']]
            sig_to_analyse = 0
        
        outPPG, peakValArr, peakIndArr, sqi, det_pattern, out_pattern, corr_pattern = AF_PPG_detector(np.array(ppg_signals[sig_to_analyse]), metadata['ppg_resample_fs'])
        # print(rest_SQI)
        if rest_SQI > metadata['ppg_sqi_lim_rest']:
            HRV_results = evaluate_heart_rate_variability(peakIndArr, sqi, metadata)
        else:
            HRV_results = {'SDNN': np.nan, 'lf_hf': np.nan, 'HR_baseline': np.nan}
    else: # set results to nan
        rest_SQI = np.nan
        HRV_results = {'SDNN': np.nan, 'lf_hf': np.nan, 'HR_baseline': np.nan}
        
    Rest_segment_analysis_results = {}
    Rest_segment_analysis_results.update({'rest_SQI': rest_SQI})
    Rest_segment_analysis_results.update(HRV_results)
    
    return Rest_segment_analysis_results