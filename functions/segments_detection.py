
import pandas as pd
import numpy as np
import os
import sys

def segments_detection(acc, metadata):
    """Find physical activity segments.

    Parameters
    ----------
    acc : df
        acceleration dataframe.
    metadata : dict
        metadata of signal.

    Returns
    -------
    df_walking_bouts : df
        DataFrame containing the informationm of physical activity segments.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from PA_algorithms import step_detection_TAF, calc_mad
    from utils import ranges

    ##  ---- Update constants
    max_stop = metadata['max_stop'] # seconds
    min_walking_dur = metadata['min_walking_duration'] #seconds
    recovery_time = metadata['recovery_time'] #seconds
    rest_time = metadata['rest_time'] #seconds

    ## Preprocess
    acc_mod_or = np.sqrt(np.sum(np.square(acc[['x','y','z']]), axis=1))
    acc_mod = np.array(acc_mod_or - np.mean(acc_mod_or))
    
    ## Step detection
    if len(acc_mod)>metadata['acc_fs']*100: # at least 100 seconds
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod, metadata['acc_fs'])
        
        steps_arr = np.array(steps_arr)
        diffs = np.diff(steps_arr/metadata['acc_fs'])
        
        where_consistent = np.where(diffs < max_stop)[0]
        stepping_ranges = np.array(ranges(where_consistent))
        ranges_dur = np.array([round(steps_arr[x[1]]/metadata['acc_fs']-steps_arr[x[0]]/metadata['acc_fs']) for x in stepping_ranges])
        
        ranges_valid = stepping_ranges[ranges_dur>min_walking_dur]
        inds_valid = np.array([np.array([int(steps_arr[x[0]]),int(steps_arr[x[1]])]) for x in ranges_valid])
        durations = ranges_dur[ranges_dur>min_walking_dur]
        steps_ranges = np.array([x[1]-x[0] for x in stepping_ranges])[ranges_dur>min_walking_dur]
        average_intensity = np.round(steps_ranges/durations,1)*60
        
        ## Find the number steps during recovery and rest
        if len(inds_valid) > 0:
            recovery_steps = [len(steps_arr[(steps_arr>x) & (steps_arr<(x + recovery_time*metadata['acc_fs']))]) for x in inds_valid[:,1]]
            rest_steps = [len(steps_arr[(steps_arr>x-rest_time*metadata['acc_fs']) & (steps_arr<(x))]) for x in inds_valid[:,0]]
            starts = inds_valid[:,0]
            ends = inds_valid[:,1]
        else: # No activity segments were detected
            recovery_steps = np.nan
            rest_steps = np.nan
            starts = np.nan
            ends = np.nan
        
        # Calculate median amplitude deviation
        mads, mn_mad = calc_mad(acc_mod, metadata['acc_fs'])
        mads_sum = np.array([np.round(np.sum(mads[int(x[0]/metadata['acc_fs']): int(x[1]/metadata['acc_fs'])]),3) for x in inds_valid])
        mads_mean = np.array([np.round(np.mean(mads[int(x[0]/metadata['acc_fs']): int(x[1]/metadata['acc_fs'])]),3) for x in inds_valid])
        
        # Create dataframe of walking segments
        dict_walking_bouts = {'ind_start': starts, 'ind_end': ends, 'duration': durations, 'steps':steps_ranges, 'mad_mean':mads_mean, 'mad_sum':mads_sum, 'intensity': average_intensity, 'recovery_steps':recovery_steps, 'rest_steps':rest_steps }
        df_walking_bouts = pd.DataFrame(dict_walking_bouts)
    else:
        df_walking_bouts = pd.DataFrame([], columns = ['ind_start', 'ind_end', 'duration', 'steps', 'mad_mean', 'mad_sum', 'intensity', 'recovery_steps', 'rest_steps'])
    
    return df_walking_bouts