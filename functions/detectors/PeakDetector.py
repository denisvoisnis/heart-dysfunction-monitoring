class PeakDetector:
    def __init__(self):
        self.sampleNo = -1
        self.tempMax = 0
        self.counter = 0
        self.maxIdx = 0
        self.maxVal = 0
        self.prevMaxIdx = 0
        self.cnt = 0
        self.inputArr = [0.0] * 3
        self.peakIdx = [0.0] * 3

    def threshold_crossing_peak_detector(self, y, Fs, numbness, threshold):
        self.sampleNo += 1
        self.peakIdx = [0.0] * 3

        if self.sampleNo >= 0 and self.sampleNo <= 2:
            self.inputArr[self.sampleNo - 1] = y
        else:
            self.inputArr[:-1] = self.inputArr[1:]
            self.inputArr[-1] = y

            if self.inputArr[1] > threshold:
                self.cnt += 1
                if self.inputArr[1] > self.tempMax and self.inputArr[1] > self.inputArr[0] and self.inputArr[1] > \
                        self.inputArr[2]:
                    self.tempMax = self.inputArr[1]
                    tempMaxIdx = self.sampleNo
                    tempMaxVal = self.tempMax
                    if self.cnt >= (0.06 * Fs):
                        self.cnt = 0
                        difference = tempMaxIdx - self.prevMaxIdx
                        if difference > (numbness * Fs) or difference == tempMaxIdx:
                            self.cnt = 0
                            self.maxIdx = tempMaxIdx
                            self.maxVal = tempMaxVal
                            self.counter = 0
            elif self.maxIdx != 0:
                self.counter += 1
                self.peakIdx[0] = self.counter
                self.peakIdx[1] = self.maxIdx
                self.peakIdx[2] = self.maxVal
                self.tempMax = 0
                self.prevMaxIdx = self.maxIdx

        return self.peakIdx