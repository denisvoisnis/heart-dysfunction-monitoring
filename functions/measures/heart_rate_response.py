# imports
import pandas as pd
import numpy as np
import os
import sys

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def estimate_fast_phase(hr_time_rec, hr_rec, window):
    """ Calculation of fast phase of heart rate response series.
    
    Parameters
    ----------
    hr_time_rec : np.array
        Time of heart rate response series.
    hr_rec : np.array
        Heart rate response series.
    window : int
        window of fast phase

    Returns
    -------
    fast_phase : float
        fast phase measure estimation.
    """
    # How many steps
    n_step = int(np.floor((hr_time_rec[-1]-hr_time_rec[0])-window))
            
    slopes = []
    for i_sec in range(n_step):
        st_ind = np.argmin(np.abs(hr_time_rec-i_sec))
        end_ind = np.argmin(np.abs(hr_time_rec-i_sec-window))
        # extract 30 seconds window
        hr_part = hr_rec[st_ind:end_ind]
        hr_time_part = hr_time_rec[st_ind:end_ind].reshape((-1, 1))
        
        model = LinearRegression().fit(hr_time_part, hr_part)
        slopes.append(model.coef_[0])
    if len(slopes) > 0: # if any slopes found
        fast_phase = np.round(slopes[np.argmin(slopes)],3) ## Estimation of fast phase
    else:
        fast_phase = np.nan
        
    return fast_phase


def estimate_slow_phase(hr_time_rec, hr_rec, HR_max):
    """ Calculation of slow phase of heart rate response series.
    
    Parameters
    ----------
    hr_time_rec : np.array
        Time of heart rate response series.
    hr_rec : np.array
        Heart rate response series.
    HR_max : float
        maximum heart rate.

    Returns
    -------
    slow_phase : float
        slow phase measure estimation.
    """
    # directories
    work_dir = '../'
    sys.path.insert(1,work_dir + '/app/functions/preprocess')
    from utils import monoExp
    
    HR_reduction = HR_max - np.median(hr_rec[-5:]) ## Estimation of slow phase
    HR_end = np.median(hr_rec[-5:])
    xVals=hr_time_rec
    yVals=hr_rec
    p0 = (HR_max, 0.1, HR_end)
    try:
        params, cv = curve_fit(monoExp, xVals, yVals, p0, maxfev=10000)
        m, t, b = params
        yvals_pol = monoExp(xVals-xVals[0], m, t, b)
        slow_phase = np.round(HR_max - yvals_pol[-1],3)
    except: # exponential cannot be adapted, HR not decreasing or too much noise
        slow_phase = np.nan
    
    return slow_phase


def estimate_heart_rate_reserve(hr_rest, age, HR_max):
    """ Calculation of heart rate reserve.
    
    Parameters
    ----------
    hr_rest : np.array
        Resting heart rate series.
    HR_max : int
        Age of subject.
    HR_max : float
        Maximum heart rate.

    Returns
    -------
    RES : float
        Heart rate reserve measure estimation.
    """
    if len(hr_rest)>10: # if rest HR found
        HR_rest = np.median(hr_rest)
        RES = np.round(((HR_max-HR_rest)/((220-age) - HR_rest))*100,3)
    else:
        RES = np.nan
    return RES


def estimate_heart_rate_response_measures(hr_rest, hr_time_rec, hr_rec, metadata):
    """ Evaluate heart rate response measures.
    
    Parameters
    ----------
    hr_rest : np.array
        array of HR rest
    hr_time_rec : np.array
        array of HR recovery time
    hr_rec : np.array
        array of HR recovery
    metadata : dict
        metadata of signal.
        
    Returns
    -------
    dict_response_measures : dict
        Dictionary of heart rate response measures.
    """
    ##
    
    ind_10_sec = np.argmin(np.abs(hr_time_rec-10))
    
    # Estimate maximum heart rate
    HR_max = np.round(np.max(hr_rec[0:ind_10_sec]),3)
    
    ## Calculate fast phase
    fast_phase = estimate_fast_phase(hr_time_rec, hr_rec, metadata['fast_phase_window'])
    
    ## Calculate slow phase 
    slow_phase = estimate_slow_phase(hr_time_rec, hr_rec, HR_max)
    
    ## Estimate heart rate reserve
    RES = estimate_heart_rate_reserve(hr_rest,  metadata['age'], HR_max)
    
    return {'HR_max':HR_max,'fast_phase':fast_phase,'slow_phase':slow_phase,'RES':RES}



def evaluate_heart_rate_response(acc, df_walking_bouts_updated, metadata):
    """ Evaluate heart rate response measures.
    
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
    dict_response_measures : dict
        Dictionary of heart rate response measures.
    """
    ##
    process = 0
    if len(df_walking_bouts_updated) > 0:
        process = 1
    
    df_HRR = pd.DataFrame([])
    
    ## Extract parameters    
    for df_wb in df_walking_bouts_updated.iterrows():
        df_wb_dat = df_wb[1]
        df_wb_dat_dict = df_wb_dat.to_dict()
        
        dict_save = {key: metadata[key] for key in ['sub_id','filename']}
        
        ind_10_sec = np.argmin(np.abs(df_wb_dat_dict['HR_rec_time_array']-10))
        if len(df_wb_dat_dict['HR_rec_array'][0:ind_10_sec])>2: ## detect max if at least 3 values in first 10 seconds
            # get segments of heart rate series
            hr_rest = df_wb_dat_dict['HR_rest_array']
            hr_time_rec = df_wb_dat_dict['HR_rec_time_array']
            hr_rec = df_wb_dat_dict['HR_rec_array']
            
            ind_10_sec = np.argmin(np.abs(hr_time_rec-10))
            
            # Estimate maximum heart rate
            HR_max = np.max(hr_rec[0:ind_10_sec])
            
            ## Calculate fast phase
            fast_phase = estimate_fast_phase(hr_time_rec, hr_rec, metadata['fast_phase_window'])
            
            ## Calculate slow phase 
            slow_phase = estimate_slow_phase(hr_time_rec, hr_rec, HR_max)
            
            ## Estimate heart rate reserve
            RES = estimate_heart_rate_reserve(hr_rest,  metadata['age'], HR_max)
            
        else: ## respnse measures cannot be estimated
            HR_max = np.nan
            RES = np.nan
            slow_phase = np.nan
            fast_phase = np.nan
        
        ## Create results dict
        dict_response_measures = {'HR_max': HR_max, 'fast_phase': fast_phase, 'slow_phase':slow_phase, 'RES':RES}
        dict_save.update({'duration': df_wb_dat_dict['duration'], 'steps':df_wb_dat_dict['steps']})
        dict_save.update(dict_response_measures)
        
        df_save = pd.DataFrame(dict_save, index = [0])
        df_HRR = pd.concat([df_HRR, df_save])
        
    return df_HRR
        
        
        
        
    
    