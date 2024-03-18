import numpy as np
from scipy.signal import convolve

class CrossCorrelation:
    def __init__(self, sig1, sig2):
        sig2_reverse = np.flip(sig2)
        self.conv = convolve(sig1, sig2_reverse, mode='full')
        self.lags = np.arange(-len(sig1) + 1, len(sig2))
        self.corr_length = len(sig1) + len(sig2) - 1
        self.is_equal_length = len(sig1) == len(sig2)
        self.sig_length = len(sig1) if self.is_equal_length else 0

    def get_frame_size(self):
        return len(self.conv)

    def get_lags(self):
        return self.lags

    def get_ccf(self):
        ccf = np.copy(self.conv)
        if self.is_equal_length:
            # normalizing like MATLAB's xcorr 'biased' option
            ccf /= self.sig_length
        return ccf