# imports
import pandas as pd
import numpy as np
import os
import sys
from scipy import interpolate, signal
from scipy.spatial import ConvexHull

def distance_walk_test(steps_min, steps_arr, metadata):
    """ Estimate walk distance.
    
    Parameters
    ----------
    steps_min : np.array
        Steps per minute array.
    steps_arr : np.array
        Array of individual steps
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    xmin_walk_test : double
        Dictionary of mobility and balance measures.
    """
    if np.isnan(metadata['step_length']): ## Aproximate 6-min walk test distance if no step length provided
        if metadata['age'] > 64:
            c1 = 0.098
            c2 = 0.0028
        else:
            c1 = 0.086
            c2 = 0.0031
        height_meters = metadata['height']/100.0
        list_of_distances = []
        for steps_n in steps_min:
            dist_n = ((c1* steps_n) + (c2 * np.power(steps_n,2))) * height_meters
            list_of_distances.append(np.round(dist_n,1))
        xmin_walk_test = np.round(np.sum(list_of_distances),2)
        
    else: ## Aproximate 6-min walk test distance if step length is provided
        xmin_walk_test = (metadata['step_length']*100) * len(steps_arr)
        
    return xmin_walk_test



def evaluate_acc_segment_steps(steps_arr_seg, metadata):
    """ Evaluate mobility and balance measures.
    
    Parameters
    ----------
    steps_arr_seg : np.array
        detected steps indexes.
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    params : dict
        Dictionary of mobility and balance measures.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    ##
    
    step_duration = np.diff(steps_arr_seg)/metadata['acc_fs']
    stride_time = np.round(np.median(step_duration)*2,3)
    
    step_duration_lim = step_duration[(step_duration<2) & (step_duration>0.3)]
    gait_std = np.round(np.std(step_duration_lim),4) # seconds
    step_duration_in_std = step_duration_lim[(step_duration_lim<(np.median(step_duration_lim)+gait_std)) & (step_duration_lim>(np.median(step_duration_lim)-gait_std))]

    gait_irregularity = np.round(np.std(step_duration_in_std),4) # seconds


    if len(step_duration)>0:
        cadence = np.round(len(step_duration),2)
    else:
        cadence = np.nan

    # diffs_analysis = np.diff(np.array(steps_arr_seg))
    diffs_analysis = step_duration_in_std
    step_r = diffs_analysis[0:-1:2]
    step_l = diffs_analysis[1:-1:2]
    len_lim = np.min([len(step_r), len(step_l)])

    if len_lim > 0:
        left_st = np.abs((step_r[0:len_lim]-step_l[0:len_lim]))
        right_st = (step_r[0:len_lim]+step_l[0:len_lim])
        
        gait_asymmetry = np.round((np.sum(np.divide(left_st,right_st))/len_lim)*100,2)
    else:
        gait_asymmetry =  np.nan
        
    dict_results = {'stride_time':stride_time,'gait_irregularity':gait_irregularity,'cadence':cadence,'gait_asymmetry':gait_asymmetry}
    return dict_results
    
    
def evaluate_acc_balance(acc_segment, metadata):
    """ Evaluate mobility and balance measures.
    
    Parameters
    ----------
    acc_segment : pd.DataFrame
        acceleration dataframe of segment. ['x','y','z']
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    params : dict
        Balance results dictionary.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    ##
    from acceleration_preprocess import acceleration_preprocess, acceleration_magnitude
    
    Acc_m_seg = acceleration_preprocess(np.array(acc_segment['y']), metadata)
    Acc_v_seg = acceleration_preprocess(np.array(acc_segment['x']), metadata)
    Acc_a_seg = acceleration_preprocess(np.array(acc_segment['z']), metadata)
    
    ## Evaluate postural sway
    points = [[x,y] for x,y in zip(Acc_a_seg, Acc_m_seg)]
    postural_sway = np.round(ConvexHull(points).area,4)
    
    ## Lissajous index
    inds_m_pos = np.where((Acc_m_seg>0) & (Acc_v_seg>0))
    inds_m_neg = np.where((Acc_m_seg<0) & (Acc_v_seg>0))
    area_n = (np.max(Acc_v_seg[inds_m_pos])-np.min(Acc_v_seg[inds_m_pos]))*np.abs(np.max(Acc_m_seg))
    area_p = (np.max(Acc_v_seg[inds_m_neg])-np.min(Acc_v_seg[inds_m_neg]))*np.abs(np.min(Acc_m_seg))
    lissajous_index = np.round(100*((2*np.abs(area_p-area_n))/(area_n+area_p)),2)
    
    ## MAD
    # Extract acceleration axis
    Acc_v = np.array(acc_segment['x'])
    Acc_m = np.array(acc_segment['y'])
    Acc_a = np.array(acc_segment['z'])
    
    # Calculate magnitude of raw acceleration signal
    acc_mod_raw = acceleration_magnitude([Acc_v, Acc_m, Acc_a])
    
    acc_mod_raw_segment = acceleration_magnitude([np.array(acc_segment['x']), np.array(acc_segment['y']), np.array(acc_segment['z'])])
    movement_vigor = np.round((np.sum(np.abs(acc_mod_raw_segment-np.mean(acc_mod_raw_segment)))/len(acc_mod_raw_segment))/1000,2)
    
    dict_results = {'postural_sway':postural_sway,'lissajous_index':lissajous_index, 'movement_vigor': movement_vigor}

    return dict_results
    


def evaluate_mobility_balance(acc, df_walking_bouts_updated, metadata):
    """ Evaluate mobility and balance measures.
    
    Parameters
    ----------
    acc : df
        acceleration dataframe.
    walking_bouts : df
        dataframe of walking bouts information
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    dict_mobility_measures : dict
        Dictionary of mobility and balance measures.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from PA_algorithms import step_detection_TAF, calc_mad
    from acceleration_preprocess import acceleration_preprocess, acceleration_magnitude, rotate_acceleration_axis, butter_lowpass_filter
    ##
    
    process = 0
    if len(df_walking_bouts_updated) > 0:
        process = 1
    
    df_mobility_balance = pd.DataFrame([])
    dict_save = {key: metadata[key] for key in ['sub_id','filename']}
    
    ## Extract parameters    
    for df_wb in df_walking_bouts_updated.iterrows():
        df_wb_dat = df_wb[1]
        df_wb_dat_dict = df_wb_dat.to_dict()
        
        ## Extract walking acceleration pattern
        acc_segment = acc.iloc[int(df_wb_dat['ind_start']):int(df_wb_dat['ind_end'])]
        acc_walking = acc_segment.iloc[metadata['acc_fs']*metadata['rest_time']:int((metadata['acc_fs']*metadata['rest_time'])+df_wb_dat['duration']*metadata['acc_fs'])]
        
        # Rotate 
        acc_original_axis = acc[['x','y','z']]
        rotated_acc = rotate_acceleration_axis(acc_original_axis)
        
        ## Calculate magnitude of mediolateral acceleration axis signal
        acc_mod_or = np.sqrt(np.sum(np.square(rotated_acc[['x']]), axis=1))
        acc_mod = np.array(acc_mod_or - np.mean(acc_mod_or))
        acc_mod_filt = butter_lowpass_filter(acc_mod, 3, 50, 5)
        steps_arr, time_array, steps_min, num_arr = step_detection_TAF(acc_mod_filt, metadata['acc_fs'], max_val = 50, min_val = 0)
        
        ## Evaluate distance made during walking
        xmin_walk_test = distance_walk_test(steps_min, steps_arr, metadata)
        
        # define segment times for the analysis of mobility and balance measures
        t_start = 30
        t_end = 60
        
        ## Extract walking segment from acceleration DataFrame
        acc_walking_segment = acc_walking.iloc[t_start*metadata['acc_fs']:t_end*metadata['acc_fs']]
        
        ## Extract steps segment from detected steps array
        steps_arr_seg = steps_arr[(steps_arr>t_start*metadata['acc_fs']) & (steps_arr<t_end*metadata['acc_fs'])]

        ## Calculate measures
        res_balance_dict = evaluate_acc_balance(acc_walking_segment, metadata)
        res_steps_results_dict = evaluate_acc_segment_steps(steps_arr_seg, metadata)
        
        ## Create results dict
        dict_mobility_measures = {'postural_sway':res_balance_dict['postural_sway'] ,'lissajous_index':res_balance_dict['lissajous_index'], 'gait_asymmetry':res_steps_results_dict['gait_asymmetry']}
        dict_mobility_measures.update({'movement_vigor':res_balance_dict['movement_vigor'], 'stride_time':res_steps_results_dict['stride_time'], 'cadence':res_steps_results_dict['cadence'], 'gait_irregularity':res_steps_results_dict['gait_irregularity'], 'xmin_walk_test':xmin_walk_test})
        
        dict_save.update({'duration': df_wb_dat_dict['duration'], 'steps':df_wb_dat_dict['steps']})
        dict_save.update(dict_mobility_measures)
        
        df_save = pd.DataFrame(dict_save, index = [0])
        df_mobility_balance = pd.concat([df_mobility_balance, df_save])
    
    return df_mobility_balance