import numpy as np
from scipy import signal
from scipy.signal import resample, correlate, resample_poly
import matplotlib.pyplot as plt
from correlation import corr
from DigitalFilter import DigitalFilter

from DigitalAdaptiveFilter import DigitalAdaptiveFilter
from PeakDetector import PeakDetector
from MedianFilter import MedianFilter
from StatisticalUtilities import StatisticalUtilities
from func_count_matches import func_count_matches
from satlin import satlin
from Resample import Resample
from CrossCorrelation import CrossCorrelation

import sys
import os

def AF_PPG_detector(rawPPGin, fs, min_corr_thresh = 0.600):
    
    mResample = Resample()

    mDigitalFilterIIR_LP_ppg = DigitalFilter()
    mDigitalFilterIIR_HP_ppg = DigitalFilter()
    mDigitalAdaptiveFilter = DigitalAdaptiveFilter()
    mPeakDetector = PeakDetector()
    mMedianFilter = MedianFilter()
    
    mStatisticalUtilitiesIntRawDiv = StatisticalUtilities()
    mStatisticalUtilitiesIntMedDiv = StatisticalUtilities()
    mStatisticalUtilitiesIntRawDiff = StatisticalUtilities()
    
    mMedianFilterCurrentInt = MedianFilter()
    mMedianFilterCurrentIntRawDiv = MedianFilter()
    mMedianFilterCurrentIntMedDiv = MedianFilter()
    mMedianFilterCurrentIntRawDiff = MedianFilter()
    
    mDigitalFilterExpIIR1 = DigitalFilter()
    mDigitalFilterExpIIR2 = DigitalFilter()
    mDigitalFilterExpIIR3 = DigitalFilter()
    
    # directories    
    script_path = os.path.dirname(os.path.realpath(__file__))

    outPPG = []
    ppgTemplateArray = []
    with open(script_path + '/assets/template_Type1_Dawber_55_250.txt') as f:
        contents = f.readlines()
        ppgTemplateArray.append(contents)
    f.close()
    #ppgTemplateArray = np.fix(ppgTemplateArray * 20)

    ppgTemplateArray = np.asfarray(ppgTemplateArray, float)
    ppgTemplateArray = [np.fix(element * 20) for element in ppgTemplateArray]
    ppgTemplateArray = ppgTemplateArray[0]
    ppgTemplateArrayTemp = ppgTemplateArray

    PLETH = [element * -1 for element in rawPPGin]
    #PLETH = rawPPGin[0]

    Fd = fs  # Sampling frequency, Hz
    peakDetectionWindow = int(2.0 * Fd)
    peakDetectionWindowOverlap = 0.9  # 0%
    peakDetectionPercentile = 0.55  # 55%

    peakDetectionThreshold = 0.0
    peakDetectionCounter = -1
    peakDetectionArray = np.zeros(peakDetectionWindow)
    sortedPeakDetectionArray = np.zeros(peakDetectionWindow)
    numbness = 0.2

    peakIdx = []  # Placeholder for peak indices array
    currentPeakIdx = 0.0
    currentPeakVal = 0.0
    peakCounter = 0

    mainIntervalCounter = 0

    ppgMorphCnt = 0
    size = 150

    pulseSize = 0

    #ppgExtractedPulseArrayList = np.zeros(size)
    ppgExtractedPulseArrayList = []

    currentIntervalRaw = 0

    totalFeatureCount = 8

    intervalArrayRaw = [0.0] * totalFeatureCount
    intervalArrayMed = [0.0] * totalFeatureCount
    intervalArrayRawDivMed = [0.0] * totalFeatureCount
    intervalArrayMedDivMed = [0.0] * totalFeatureCount
    intervalArrayRawDiff = [0.0] * totalFeatureCount
    intervalArrayRawDiffMed = [0.0] * totalFeatureCount

    sqiWindow = 2

    corrValArray = [0.0] * sqiWindow
    corrLagArray = [0.0] * sqiWindow

    peakValWindow = 4
    peakValArray = [0.0] * peakValWindow

    minCorrVal = 0.0
    maxCorrLag = 0.0
    meanPeakVal = 0.0

    alfa = 0.02
    b = [alfa ** 2]
    a = [1.0, -2.0 * (1 - alfa), (1 - alfa) ** 2]

    nDiff = 0.0
    detectorDecis = 0
    detectorDecis2 = 0

    Q = 0.0
    arrhytCnt = 0
    nRMS = 0.0
    nDiffLP = 0.0
    nMean = 0.0
    diffDivRms = 0.0
    diffDivMean = 0.0

    artCnt = 0
    ampCnt = 0

    # LP filter coefficients
    B_LP = [3.649246226465329e-04, 0.001459698490586, 0.002189547735879, 0.001459698490586, 3.649246226465329e-04]
    A_LP = [1.0, -3.419753240513017, 4.526774177988911, -2.738063526698766, 0.637227353478029]

    # HP filter coefficients
    B_HP = [0.930170138647645, -1.860340277295290, 0.930170138647645]
    A_HP = [1.0, -1.970251633832550, 0.970893103812223]

    peakValArr = []
    peakIndArr = []

    det_pattern = []
    out_pattern = []
    corr_pattern = []
    sqi = []

    for y in PLETH:
        # Processing with FIR High-pass filter
        y = mDigitalFilterIIR_HP_ppg.IIRFilter(y, B_HP, A_HP)
        # Processing with FIR Low-pass filter
        y = mDigitalFilterIIR_LP_ppg.IIRFilter(y, B_LP, A_LP)
        # Perform baseline wander removal using NLMS adaptive filter
        y = mDigitalAdaptiveFilter.NLMS(0.015, 5, y, 1)

        peakDetectionCounter += 1

        if peakDetectionCounter < int(peakDetectionWindow - (peakDetectionWindow * peakDetectionWindowOverlap)):
            peakDetectionArray[int(peakDetectionCounter + (peakDetectionWindow * peakDetectionWindowOverlap))] = y
        else:
            sortedPeakDetectionArray = np.sort(peakDetectionArray)
            peakDetectionThreshold = sortedPeakDetectionArray[int(peakDetectionWindow * peakDetectionPercentile) - 1]

            if peakDetectionWindowOverlap != 0:
                peakDetectionArray[:int(peakDetectionWindow * peakDetectionWindowOverlap)] = peakDetectionArray[
                                                                                             int(peakDetectionWindow - peakDetectionWindow * peakDetectionWindowOverlap):]

            peakDetectionCounter = -1

        # Positive peak detection
        peakIdx = mPeakDetector.threshold_crossing_peak_detector(y, Fd, numbness, peakDetectionThreshold)

        if peakIdx[0] == 1:
            peakCounter += 1

        # PPG pulse extraction
        if ppgMorphCnt > size - 1:
            ppgMorphCnt = size - 1
            ppgExtractedPulseArrayList.insert(ppgMorphCnt, y)
        else:
            ppgExtractedPulseArrayList.insert(ppgMorphCnt, y)

        ppgMorphCnt += 1

        if peakCounter == 1 and peakIdx[1] != 0:
            currentPeakIdx = peakIdx[1]
            currentPeakVal = peakIdx[2]
        elif peakCounter == 2:
            peakCounter = 1

            mainIntervalCounter += 1

            if ppgMorphCnt > size - 1:
                pulseSize = size
            else:
                pulseSize = ppgMorphCnt

            # Current interval convention from samples to seconds
            currentIntervalRaw = (peakIdx[1] - currentPeakIdx) / Fd
            currentIntervalMed = mMedianFilterCurrentInt.median_filter(currentIntervalRaw, 3)

            currentIntervalRawDiv = mStatisticalUtilitiesIntRawDiv.div(currentIntervalRaw)
            currentIntervalMedDiv = mStatisticalUtilitiesIntMedDiv.div(currentIntervalMed)

            currentIntervalRawDivMed = mMedianFilterCurrentIntRawDiv.median_filter(currentIntervalRawDiv, 3)
            currentIntervalMedDivMed = mMedianFilterCurrentIntMedDiv.median_filter(currentIntervalMedDiv, 3)

            currentIntervalRawDiff = mStatisticalUtilitiesIntRawDiff.diff(currentIntervalRaw)
            currentIntervalRawDiffMed = mMedianFilterCurrentIntRawDiff.median_filter(currentIntervalRawDiff, 3)

            # TEMPLATE MATCHING

            ppgExtractedPulseArray = np.zeros(pulseSize)

            for i in range(pulseSize):
                ppgExtractedPulseArray[i] = ppgExtractedPulseArrayList[i]

            ppgExtractedPulseArrayList.clear()

            # Resample PPG template to match current PPG pulse samples
            ppgTemplateResampledArray = resample(ppgTemplateArray, pulseSize)
            #ppgTemplateResampledArray = resample_poly(ppgTemplateArray, len(ppgExtractedPulseArrayList), len(ppgTemplateArray))

            #ppgTemplateResampledArray = mResample.resample(ppgTemplateArray, len(ppgTemplateArray), False, len(ppgTemplateArray), len(ppgExtractedPulseArrayList))

            # Normalize resampled template and extracted pulse by mean and standard deviation
            ppgTemplateResampledArray = (ppgTemplateResampledArray - np.mean(ppgTemplateResampledArray)) / np.std(
                ppgTemplateResampledArray)
            ppgExtractedPulseArray = (ppgExtractedPulseArray - np.mean(ppgExtractedPulseArray)) / np.std(
                ppgExtractedPulseArray)

            # Calculation of normalized cross-correlation function
            # ccfResult = correlate(ppgExtractedPulseArrayList, ppgTemplateResampledArray, mode='full', method='direct')
            # ccfResult /= len(ppgExtractedPulseArrayList)
            # ccfLags = np.arange(-len(ppgExtractedPulseArrayList) + 1, len(ppgExtractedPulseArrayList))
            #
            # currentCorrMaxVal = np.max(ccfResult)
            # currentCorrMaxLagIdx = np.argmax(ccfResult)
            # currentCorrMaxLag = ccfLags[currentCorrMaxLagIdx] / Fd

            mCrossCorrelation = CrossCorrelation(ppgExtractedPulseArray, ppgTemplateResampledArray)
            # Get the cross-correlation function (CCF)
            ccf = mCrossCorrelation.get_ccf()

            ccfLags = mCrossCorrelation.get_lags()

            currentCorrMaxVal = 0
            currentCorrMaxLagIdx = 0

            for k in range(len(ccf)):
                if ccf[k] > currentCorrMaxVal:
                    currentCorrMaxVal = ccf[k]
                    currentCorrMaxLagIdx = k

            currentCorrMaxLag = ccfLags[currentCorrMaxLagIdx] / Fd

            ppgMorphCnt = 0
            ppgExtractedPulseArrayList= []

            if currentCorrMaxVal > 0.78 and (-0.10 < currentCorrMaxLag < 0.10) and meanPeakVal > 200:
                numbness = currentIntervalMed * 0.45
                if numbness > 0.20:
                    numbness = 0.20

            if currentCorrMaxVal > 0.95 and (-0.05 < currentCorrMaxLag < 0.05):
                ppgTemplateArray = ppgExtractedPulseArray
            else:
            #     ppgTemplateArray = ppgTemplateArrayTemp
                  ppgTemplateArray = [np.fix(element * 20) for element in ppgTemplateArrayTemp]

            if 1 <= mainIntervalCounter <= totalFeatureCount:
                intervalArrayRaw[mainIntervalCounter - 1] = currentIntervalRaw
                intervalArrayMed[mainIntervalCounter - 1] = currentIntervalMed
                intervalArrayRawDivMed[mainIntervalCounter - 1] = currentIntervalRawDivMed
                intervalArrayMedDivMed[mainIntervalCounter - 1] = currentIntervalMedDivMed
                intervalArrayRawDiff[mainIntervalCounter - 1] = currentIntervalRawDiff
                intervalArrayRawDiffMed[mainIntervalCounter - 1] = currentIntervalRawDiffMed
            elif mainIntervalCounter > totalFeatureCount:
                intervalArrayRaw[:-1] = intervalArrayRaw[1:]
                intervalArrayRaw[totalFeatureCount - 1] = currentIntervalRaw

                intervalArrayMed[:-1] = intervalArrayMed[1:]
                intervalArrayMed[totalFeatureCount - 1] = currentIntervalMed

                intervalArrayRawDivMed[:-1] = intervalArrayRawDivMed[1:]
                intervalArrayRawDivMed[totalFeatureCount - 1] = currentIntervalRawDivMed

                intervalArrayMedDivMed[:-1] = intervalArrayMedDivMed[1:]
                intervalArrayMedDivMed[totalFeatureCount - 1] = currentIntervalMedDivMed

                intervalArrayRawDiff[:-1] = intervalArrayRawDiff[1:]
                intervalArrayRawDiff[totalFeatureCount - 1] = currentIntervalRawDiff

                intervalArrayRawDiffMed[:-1] = intervalArrayRawDiffMed[1:]
                intervalArrayRawDiffMed[totalFeatureCount - 1] = currentIntervalRawDiffMed

            if sqiWindow > 1:
                if 1 <= mainIntervalCounter <= sqiWindow:
                    corrValArray[mainIntervalCounter - 1] = currentCorrMaxVal
                    corrLagArray[mainIntervalCounter - 1] = currentCorrMaxLag

                    minCorrVal = currentCorrMaxVal
                    maxCorrLag = abs(currentCorrMaxLag)
                elif mainIntervalCounter > sqiWindow:
                    corrValArray[:-1] = corrValArray[1:]
                    corrValArray[sqiWindow - 1] = currentCorrMaxVal

                    corrLagArray[:-1] = corrLagArray[1:]
                    corrLagArray[sqiWindow - 1] = currentCorrMaxLag

                    minCorrVal = min(corrValArray)
                    # maxCorrLag = max(abs(corrLagArray))
                    maxCorrLag = max(abs(lag) for lag in corrLagArray)
            else:
                minCorrVal = currentCorrMaxVal
                maxCorrLag = abs(currentCorrMaxLag)

            if peakValWindow > 1:
                if 1 <= mainIntervalCounter <= peakValWindow:
                    peakValArray[mainIntervalCounter - 1] = currentPeakVal
                    meanPeakVal = currentPeakVal
                elif mainIntervalCounter > peakValWindow:
                    peakValArray[:-1] = peakValArray[1:]
                    peakValArray[peakValWindow - 1] = currentPeakVal
                    meanPeakVal = np.mean(peakValArray)
            else:
                meanPeakVal = currentPeakVal

            if minCorrVal > min_corr_thresh and meanPeakVal > 200:
                mDiff = StatisticalUtilities()
                diffs = [mDiff.diff(interval) for interval in intervalArrayRaw]

                cross = 0
                tempCross = 0
                crossCounter = 0
                for i in range(1, totalFeatureCount):
                    if diffs[i] > 0:
                        cross = 1
                    else:
                        cross = 0

                    if i > 1:
                        if cross - tempCross != 0:
                            crossCounter += 1
                    tempCross = cross

                diffCnt = 0
                for i in range(1, totalFeatureCount - 1):
                    if (intervalArrayRawDiff[i - 1] > 0.15 and intervalArrayRawDiffMed[i - 1] > 0.15) or \
                            (intervalArrayRawDiff[i - 1] < -0.15 and intervalArrayRawDiffMed[i - 1] < -0.15):
                        if (intervalArrayRawDiff[i] < -0.15 and intervalArrayRawDiffMed[i] < -0.15) or \
                                (intervalArrayRawDiff[i] > 0.15 and intervalArrayRawDiffMed[i] > 0.15):
                            if (intervalArrayRawDiff[i + 1] > 0.15 and intervalArrayRawDiffMed[i + 1] > 0.15) or \
                                    (intervalArrayRawDiff[i + 1] < -0.15 and intervalArrayRawDiffMed[i + 1] < -0.15):
                                diffCnt += 1

                if crossCounter < 2 or diffCnt > 2:
                    nDiff = 0
                else:
                    nDiff = func_count_matches(intervalArrayMed, intervalArrayRawDiffMed, totalFeatureCount, 0.03)

                rSum = 0.0001
                rmSum = 0.0001
                for i in range(totalFeatureCount):
                    rmSum += intervalArrayMedDivMed[i]
                    rSum += intervalArrayRawDivMed[i]

                nRMS = ((rmSum / rSum) - 1) ** 2

                nDiffLP = mDigitalFilterExpIIR1.IIRFilter(nDiff, b, a)
                nMean = mDigitalFilterExpIIR2.IIRFilter(satlin(currentIntervalMed), b, a)
                diffDivRms = mDigitalFilterExpIIR3.IIRFilter(nRMS, b, a)

                diffDivMean = nDiffLP / nMean

                if diffDivRms < 0.0004:
                    diffDivMean = diffDivRms

                if diffDivMean < 0.630:
                    detectorDecis = 0
                else:
                    arrhytCnt += 1
                    detectorDecis = 1

                Q = 1
            else:
                artCnt += 1
                Q = 0

                if artCnt >= 2:
                    artCnt = 0
                    for i in range(6, totalFeatureCount):
                        intervalArrayRaw[i] = 0
                        intervalArrayMed[i] = 0
                        intervalArrayRawDivMed[i] = 0
                        intervalArrayMedDivMed[i] = 0
                        intervalArrayRawDiff[i] = 0
                        intervalArrayRawDiffMed[i] = 0

            peakValArr.append(currentPeakVal)
            peakIndArr.append(currentPeakIdx)

        det_pattern.append(detectorDecis)
        out_pattern.append(diffDivMean)
        corr_pattern.append(minCorrVal)
        sqi.append(Q)
        outPPG.append(y)

    return outPPG, peakValArr, peakIndArr, sqi, det_pattern, out_pattern, corr_pattern
