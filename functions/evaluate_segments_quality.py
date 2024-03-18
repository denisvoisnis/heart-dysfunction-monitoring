# imports
import pandas as pd
import numpy as np
import os
import sys


def extract_best_quality_ppg(ppg_segment, metadata, ind_ppg_rec):
    """Analyse signal - find segments.
    
    Parameters
    ----------
    ppg_segment : df
        Photoplethymogram dataframe segment.
    metadata : dict
        metadata of signal.
    ind_ppg_rec : int
        Index of the start of recovery
        
    Returns
    -------
    sqi_quality : float
        Quality of ppg segment.
    ind_arr : np.array
        Extracted pp intervals of ppg signal.
    """
    
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/detectors')
    from AF_PPG_detector import AF_PPG_detector
    
    ppg_signals = [ppg_segment[x] for x in ppg_segment.keys() if 'ppg' in x]
    sqis = []
    sqis_perc = []
    for n_sig in ppg_signals:
        outPPG, peakValArr, peakIndArr, sqi, det_pattern, out_pattern, corr_pattern = AF_PPG_detector(np.array(n_sig), metadata['ppg_fs'])
        sqi_recovery = sqi[ind_ppg_rec:]
        sqis.append(np.round(np.sum(sqi_recovery)/len(sqi_recovery),3))
    
    ## get best quality signal
    signal_to_analyse = ppg_signals[np.argmax(sqis)]
    sqi_quality = sqis[np.argmax(sqis)]
    
    outPPG_n, peakValArr_n, peakIndArr_n, sqi_n, det_pattern_n, out_pattern_n, corr_pattern_n = AF_PPG_detector(np.array(signal_to_analyse), metadata['ppg_fs'])
    
    sqi_arr = np.array(sqi_n)[peakIndArr_n]
    sqi_arr_shift = sqi_arr * np.roll(sqi_arr, -1)
    ind_arr = np.array(peakIndArr_n)[np.array(sqi_arr_shift).astype(bool)]
    return sqi_quality, ind_arr




def evaluate_segments_quality_ecg(acc, rr, df_walking_bouts, metadata):
    """Analyse signal - find segments.
    
    Parameters
    ----------
    acc : df
        acceleration dataframe.
    rr : df
        rr intervals dataframe.
    walking_bouts : df
        dataframe of walking bouts information with ppg quality
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    df_walking_bouts_updated : df
        DataFrame containing the informationm of physical activity segemnts.
    """
    
    # ----------- Evaluate segments
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from remove_ectopic_beats import detect_ectopics
    
    lim_steps_recovery = metadata['max_walking_steps_recovery']
    minimal_walk_test_duration = metadata['minimal_walk_test_duration']
    
    ## Filter by limits of walking bouts for recovery
    df_walking_bouts_use = df_walking_bouts.loc[(df_walking_bouts['recovery_steps'] < lim_steps_recovery) & (df_walking_bouts['duration'] > minimal_walk_test_duration)]
    
    df_walking_bouts_updated = pd.DataFrame([])
    
    
    for df_wb in df_walking_bouts_use.iterrows(): # .iloc[1:2]
        df_wb_dat = df_wb[1]
        df_wb_dat_dict = df_wb_dat.to_dict()
        
        ## extract signals of segment
        acc_seg = acc.iloc[int(df_wb_dat['ind_start']-50*metadata['rest_time']):int(df_wb_dat['ind_end']+50*metadata['recovery_time'])]
        time_start = acc_seg.iloc[0]['timestamp']
        time_end = acc_seg.iloc[-1]['timestamp']
        
        # normalize timestamps
        acc_seg = acc_seg.assign(timestamp=np.array(acc_seg['timestamp'])-acc_seg['timestamp'].iloc[0])
        
        ### Extract rr of recovery
        ind_start_rr = np.argmin(np.abs(rr['timestamp']-time_start))
        ind_end_rr = np.argmin(np.abs(rr['timestamp']-time_end))
        
        rr_seg = rr.iloc[ind_start_rr:ind_end_rr]
        rr_seg = rr_seg.assign(timestamp = rr_seg['timestamp']-rr_seg['timestamp'].iloc[0])
        
        ind_rr_rest = np.argmin(np.abs(np.array(rr_seg['timestamp']) - metadata['rest_time']))
        ind_rr_rec = np.argmin(np.abs(np.array(rr_seg['timestamp']) - metadata['rest_time'] - df_wb_dat['duration']))
        

        RR_time = np.array(rr['timestamp'] - rr['timestamp'].iloc[-1])
        RR_sec = rr['rrms']/1000
        HR = np.divide(60,RR_sec)
        
        HR_good_inds = np.where((HR<200) & (HR>40))[0]     
        HR_clean = np.round(HR[HR_good_inds],5)
        time_clean = np.round(RR_time[0:][HR_good_inds],3)
        time_clean = time_clean - time_clean[0]
        
        ind_recovery_start = np.argmin(np.abs(time_clean-(df_wb_dat['duration'])-metadata['rest_time']))
        ind_rest_start = np.argmin(np.abs(time_clean-metadata['rest_time']))
        
        
        HR_recovery = np.array(HR_clean[ind_recovery_start:])
        time_recovery = np.array(time_clean[ind_recovery_start:]-time_clean[ind_recovery_start])
        
        HR_rest = np.array(HR_clean[0:ind_rest_start])
        time_rest = np.array(time_clean[0:ind_rest_start]-time_clean[0])
        
        
        ### Clean RR
        ## Remove ectopic and false beats of HR recovery series
        bad_beats_recovery = detect_ectopics(HR_recovery)
        HR_recovery_removed = np.delete(HR_recovery, bad_beats_recovery)
        time_recovery_removed = np.delete(time_recovery, bad_beats_recovery)
        ## Remove ectopic and false beats of resting HR series
        bad_beats_rest = detect_ectopics(HR_rest)
        HR_rest_removed = np.delete(HR_rest, bad_beats_rest)
        time_rest_removed = np.delete(time_rest, bad_beats_rest)
        
        
        
        # Save info to dict
        dict_ecg_quality = {'HR_rec_array':[HR_recovery_removed], 'HR_rec_time_array':[time_recovery_removed], 'HR_rest_array':[HR_rest_removed], 'HR_rest_time_array':[time_rest_removed]}
        df_wb_dat_dict.update(dict_ecg_quality)
        
        df_save = pd.DataFrame(df_wb_dat_dict, index = [0])
        df_walking_bouts_updated = pd.concat([df_walking_bouts_updated, df_save])
        
    return df_walking_bouts_updated
        
        
    

def evaluate_segments_quality_ppg(acc, ppg, df_walking_bouts, metadata):
    """Analyse signal - find segments.
    
    Parameters
    ----------
    acc : df
        acceleration dataframe.
    ppg : df
        photoplethymogram dataframe.
    walking_bouts : df
        dataframe of walking bouts information with ppg quality
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    df_walking_bouts_updated : df
        DataFrame containing the informationm of physical activity segemnts.
    """
    
    # ----------- Evaluate segments
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from remove_ectopic_beats import detect_ectopics
    
    lim_steps_recovery = metadata['max_walking_steps_recovery']
    minimal_walk_test_duration = metadata['minimal_walk_test_duration']
    
    ## Filter by limits of walking bouts for recovery
    df_walking_bouts_use = df_walking_bouts.loc[(df_walking_bouts['recovery_steps'] < lim_steps_recovery) & (df_walking_bouts['duration'] > minimal_walk_test_duration)]
    
    df_walking_bouts_updated = pd.DataFrame([])
    
    for df_wb in df_walking_bouts_use.iterrows(): # .iloc[1:2]
        df_wb_dat = df_wb[1]
        df_wb_dat_dict = df_wb_dat.to_dict()
        
        ## extract signals of segment
        acc_seg = acc.iloc[int(df_wb_dat['ind_start']-50*metadata['rest_time']):int(df_wb_dat['ind_end']+50*metadata['recovery_time'])]
        time_start = acc_seg.iloc[0]['timestamp']
        time_end = acc_seg.iloc[-1]['timestamp']
        
        # normalize timestamps
        acc_seg = acc_seg.assign(timestamp=np.array(acc_seg['timestamp'])-acc_seg['timestamp'].iloc[0])
        
        # Extract indexes and segments
        ind_start_ppg = np.argmin(np.abs(ppg['timestamp']-time_start))
        ind_end_ppg = np.argmin(np.abs(ppg['timestamp']-time_end))
        
        ppg_seg = ppg.iloc[ind_start_ppg:ind_end_ppg]
        ppg_seg = ppg_seg.assign(timestamp = ppg_seg['timestamp']-ppg_seg['timestamp'].iloc[0])
        
        ind_ppg_rest = np.argmin(np.abs(np.array(ppg_seg['timestamp']) - metadata['rest_time']))
        ind_ppg_rec = np.argmin(np.abs(np.array(ppg_seg['timestamp']) - metadata['rest_time'] - df_wb_dat['duration']))

        ## evaluate quality of ppg available signals
        sqi_quality, ind_arr = extract_best_quality_ppg(ppg_seg, metadata, ind_ppg_rec)
        
        with np.errstate(divide='ignore'):
            pp_sec = (np.diff(np.array(ind_arr))/135)
            HR = np.divide(60,pp_sec)
        HR_good_inds = np.where((HR<200) & (HR>40))[0]     
        HR_clean = np.round(HR[HR_good_inds],3)
        time_clean = np.round((np.array(ind_arr[1:])/135)[HR_good_inds],3)
        
        ## segment HR series to recovery and rest
        ind_recovery_start = np.argmin(np.abs(time_clean-(df_wb_dat['duration'])-metadata['rest_time']))
        ind_rest_start = np.argmin(np.abs(time_clean-metadata['rest_time']))
        
        HR_recovery = HR_clean[ind_recovery_start:]
        time_recovery = time_clean[ind_recovery_start:]-time_clean[ind_recovery_start]
        
        HR_rest = HR_clean[0:ind_rest_start]
        time_rest = time_clean[0:ind_rest_start]-time_clean[0]
        
        
        ## Remove ectopic and false beats of HR recovery series
        bad_beats_recovery = detect_ectopics(HR_recovery)
        HR_recovery_removed = np.delete(HR_recovery, bad_beats_recovery)
        time_recovery_removed = np.delete(time_recovery, bad_beats_recovery)
        
        ## Remove ectopic and false beats of resting HR series
        bad_beats_rest = detect_ectopics(HR_rest)
        HR_rest_removed = np.delete(HR_rest, bad_beats_rest)
        time_rest_removed = np.delete(time_rest, bad_beats_rest)
        
        
        # Save info to dict
        dict_ppg_quality = {'SQI_recovery': sqi_quality, 'HR_rec_array':[HR_recovery_removed], 'HR_rec_time_array':[time_recovery_removed], 'HR_rest_array':[HR_rest_removed], 'HR_rest_time_array':[time_rest_removed]}
        df_wb_dat_dict.update(dict_ppg_quality)
        
        df_save = pd.DataFrame(df_wb_dat_dict, index = [0])
        df_walking_bouts_updated = pd.concat([df_walking_bouts_updated, df_save])
        
    return df_walking_bouts_updated