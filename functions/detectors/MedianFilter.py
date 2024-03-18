import numpy as np


class MedianFilter:
    def __init__(self):
        self.in_buffer = None
        self.flag = True

    def median_filter(self, input_value, med_filt_ord):
        if self.flag:
            self.in_buffer = [0.0001] * med_filt_ord
            self.flag = False

        # Shift the buffer and update the last element
        self.in_buffer[:-1] = self.in_buffer[1:]
        self.in_buffer[-1] = input_value

        return self.calculate_median(self.in_buffer)

    @staticmethod
    def calculate_median(array):
        sorted_array = np.sort(array)
        length = len(sorted_array)

        if length % 2 == 0:
            return (sorted_array[(length // 2) - 1] + sorted_array[length // 2]) / 2
        else:
            return sorted_array[length // 2]