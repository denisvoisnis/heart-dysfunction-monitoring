# -*- coding: utf-8 -*-
# imports
import numpy as np
import sys
from scipy import signal

def evaluate_physical_activity(acc, metadata):
    """ Evaluate physical activity measures.

    Parameters
    ----------
    acc : df
        acceleration dataframe.
    metadata : dict
        metadata of signal.

    Returns
    -------
    dict_PA_measures : dict
        Dictionary containing physical activity measures.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from PA_algorithms import step_detection_TAF
    from utils import ranges
    from acceleration_preprocess import acceleration_preprocess, acceleration_magnitude, rotate_acceleration_axis, butter_lowpass_filter
    
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
        acc_mod_or = rotated_acc['x']
        acc_mod = np.array(acc_mod_or - np.mean(acc_mod_or))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, metadata['acc_fs'], 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 50, min_val = 0)     # fs_oh1, max_val = 100, min_val = 0
    
    if position_configuration == 2:
        acc_mod_or = acceleration_magnitude([rotated_acc['x'], rotated_acc['y'], rotated_acc['z']])
        acc_mod = (np.array(acc_mod_or - np.mean(acc_mod_or)))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, metadata['acc_fs'], 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 50, min_val = 0)   # , max_val = 150, min_val = 0
    
    step_count = np.sum(steps_min)
    active_minutes =  len(np.where(steps_min>metadata['active_minute_steps'])[0])
    sedentary_time =  len(np.where(steps_min<metadata['active_minute_steps']+1)[0])
    total_minutes = len(steps_min)
    
    # Walking bouts
    walking_minutes = steps_min.copy()
    walking_minutes[walking_minutes<metadata['active_minute_steps']+1] = 0
    walking_minutes[walking_minutes>metadata['active_minute_steps']] = 1
    ranges_walking_bouts = ranges(np.where(walking_minutes==1)[0])
    durations_walking_bouts = [(x[1]-x[0])+1 for x in ranges_walking_bouts]
    
    if len(durations_walking_bouts) > 0: # if any walking bouts detected
        longest_walking_bout = np.max(durations_walking_bouts)
        mean_walking_bout = np.round(np.mean(durations_walking_bouts),2)
    else:
        longest_walking_bout = np.nan
        mean_walking_bout = np.nan
        
    ## Create results dict
    dict_PA_measures = {'step_count':step_count ,'active_minutes':active_minutes, 'longest_walking_bout':longest_walking_bout, 'mean_walking_bout':mean_walking_bout, 'sedentary_time':sedentary_time, 'total_minutes':total_minutes}
    return dict_PA_measures
    
    