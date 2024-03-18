# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import sys
import os

directory = os.path.dirname(__file__)
sys.path.insert(1,directory + '/functions/')
sys.path.insert(1,directory + '/functions/preprocess/')
sys.path.insert(1,directory + '/functions/measures/')
sys.path.insert(1,directory + '/functions/detectors/')

os.chdir(directory)
        
from segments_detection import segments_detection 
from rest_segments_detection import rest_segments_detection
from evaluate_segments_quality import evaluate_segments_quality_ppg

from mobility_balance import evaluate_mobility_balance
from physical_activity import evaluate_physical_activity
from heart_rate_response import evaluate_heart_rate_response

class heart_PA_ppg(object):
    """ Class for evaluation of ppg and acc signals.
    
    """
    def __init__(self, acc_, ppg_, metadata_):
        ## init metedata
        self.metadata = metadata_
        
        analysis_specific_durations = {'recovery_time':60,'rest_time':30,'max_stop':5,'min_walking_duration':60,'baseline_rest_duration':1}
        analysis_specific_limits = {'minimal_walk_test_duration':300,'max_walking_steps_recovery':16,'active_minute_steps':60,'max_steps_during_rest':5,'fast_phase_window':30}
        analysis_specific_sqi = {'ppg_sqi_lim':0.8, 'ppg_sqi_lim_rest': 0.8, 'ppg_resample_fs': 100}

        self.metadata.update(analysis_specific_durations)
        self.metadata.update(analysis_specific_limits)    
        self.metadata.update(analysis_specific_sqi)    
        
        ## add default metadata parameters if not specified
        #'position' - ['wrist','chest']

        ## init signals
        self.acc = acc_
        self.ppg = ppg_
        ## Create starting dict
        self.dict_sig = {}
        
        self.signals_present = 0 
        
        ## create dummy results DataFrame
        self.df_sig_res = pd.DataFrame([])
        self.df_wb_res = pd.DataFrame([])
        
        ### Check inputs
        self.check_inputs()
        
    def check_inputs(self):
        """ Check if signal inputs are correct datatype.
        
        """
        if isinstance(self.acc, (pd.DataFrame)) and isinstance(self.ppg, (pd.DataFrame)):
            self.keys_acc = self.acc.keys()
            self.keys_ppg = self.ppg.keys()
            
            ## Find PPG signals number
            ppg_n = len([x for x in self.ppg.keys() if 'ppg' in x])
            dict_ppg_settings = {'ppg_n':ppg_n}
            self.metadata.update(dict_ppg_settings)   
            
            self.signals_present = 1
        else:
            pass
    
    def perform_analysis(self):
        """ Perform analysis on signals.
        
        """
        self.analysis_performed_flag = 0
        if self.signals_present == 1:
            ## Detect segments
            df_walking_bouts = segments_detection(self.acc, self.metadata)
            
            ## Evaluate quality
            self.df_walking_bouts_quality = evaluate_segments_quality_ppg(self.acc, self.ppg, df_walking_bouts, self.metadata)
            
            ## Estimate physical activity
            self.PA_measures = evaluate_physical_activity(self.acc, self.metadata)
            
            ## Detect rest segments/ Estimate HRV measures
            self.HRV_measures = rest_segments_detection(self.acc, self.ppg, self.metadata)
            
            ## Estimate mobility/balance measures
            self.df_mobility_balance = evaluate_mobility_balance(self.acc, self.df_walking_bouts_quality, self.metadata)
            
            ## Estimate response measures
            self.df_HRR = evaluate_heart_rate_response(self.acc, self.df_walking_bouts_quality, self.metadata)
            
            self.analysis_performed_flag = 1
        else:
            pass
        
        if self.analysis_performed_flag == 1:
            self.combine_results_to_dataframe()
        else:
            pass
        
        return self.df_sig_res, self.df_wb_res
            
    def combine_results_to_dataframe(self):
        """ Combine results to dataframe.
        
        """
        ## Construct PA and HRV measures DataFrame
        self.dict_save = {key: self.metadata[key] for key in ['sub_id','filename']}
        self.dict_save.update(self.HRV_measures)
        self.dict_save.update(self.PA_measures)
        self.df_sig_res = pd.DataFrame(self.dict_save, index = [0])
        
        ## Construct walking bouts estimation dataFrame
        if len(self.df_mobility_balance) > 0:
            self.df_wb_res = self.df_HRR.merge(self.df_mobility_balance, left_on=['sub_id','filename','duration', 'steps'], right_on=['sub_id','filename','duration', 'steps'])
        else:
            self.df_wb_res = pd.DataFrame([])
        
        
        