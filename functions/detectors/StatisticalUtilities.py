import numpy as np


class StatisticalUtilities:
    def __init__(self):
        self.flag_diff = True
        self.flag_div = True
        self.temp_input_diff = 0
    
    def diff(self, input_value):
        if self.flag_diff:
            self.temp_input_diff = input_value
            self.flag_diff = False
        
        output = input_value - self.temp_input_diff
        self.temp_input_diff = input_value
        
        return output
    
    def div(self, input_value):
        if self.flag_div:
            self.temp_input_div = input_value
            self.flag_div = False
        
        if self.temp_input_div == 0:
            self.temp_input_div = 1
        
        output = input_value / self.temp_input_div
        self.temp_input_div = input_value
        
        return output
    
    def mean_var_std(self, input_array):
        mean = 0
        M = 0
        
        for n in range(1, len(input_array) + 1):
            delta = input_array[n - 1] - mean
            mean = mean + delta / n
            M = M + delta * (input_array[n - 1] - mean)
        
        var = M / (n - 2)
        std = np.sqrt(M / (n - 2))
        
        return mean, var, std