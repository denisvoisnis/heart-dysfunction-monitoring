import sys
import pandas as pd
from scipy import signal
import os.path
from os import path
import numpy as np

def detect_ectopics(hr_c):
    
    dRRs = hr_c[1:] - hr_c[0:-1]
    dRRs = np.append(np.array([0]), dRRs) 
    
    Th1 = np.empty(len(dRRs))
    for i_th in range(len(dRRs)):
        ## get drrs array
        if i_th < 45:
            lim_min = 0
        else:
            lim_min = i_th-45
        if i_th > len(dRRs)-45:
            lim_max = len(dRRs)
        else:
            lim_max = i_th+45
        drrs_array = np.abs(dRRs[lim_min:lim_max])
        Q1 = np.quantile(drrs_array, 0.25)
        Q3 = np.quantile(drrs_array, 0.75)
        QD = (Q3 - Q1)/2
        Th1[i_th] = 5.2 * QD
    
    with np.errstate(divide='ignore'):
        dRR = dRRs / Th1
    
    mRRs = np.empty(len(hr_c))
    for i_mrr in range(len(hr_c)):
        if i_mrr < 5:
            lim_min = 0
        else:
            lim_min = i_mrr-5
        if i_mrr > len(hr_c)-5:
            lim_max = len(hr_c)
        else:
            lim_max = i_mrr+5
        rr_array = np.abs(hr_c[lim_min:lim_max])
        mRRsi = hr_c[i_mrr] - np.median(rr_array)
        if mRRsi < 0:
            mRRs[i_mrr] = mRRsi*2
        else:
            mRRs[i_mrr] = mRRsi
    
    medRR = np.empty(len(hr_c))
    for i_mrr in range(len(hr_c)):
        if i_mrr < 5:
            lim_min = 0
        else:
            lim_min = i_mrr-5
        if i_mrr > len(hr_c)-5:
            lim_max = len(hr_c)
        else:
            lim_max = i_mrr+5
        rr_array = np.abs(hr_c[lim_min:lim_max])
        medRR[i_mrr] = hr_c[i_mrr] - np.median(rr_array)
    
    
    
    Th2 = np.empty(len(mRRs))
    for i_th2 in range(len(mRRs)):
        ## get drrs array
        if i_th2 < 45:
            lim_min = 0
        else:
            lim_min = i_th2-45
        if i_th2 > len(mRRs)-45:
            lim_max = len(mRRs)
        else:
            lim_max = i_th2+45
        mrrs_array = np.abs(mRRs[lim_min:lim_max])
        Q1 = np.quantile(mrrs_array, 0.25)
        Q3 = np.quantile(mrrs_array, 0.75)
        QD = (Q3 - Q1)/2
        Th2[i_th2] = 5.2 * QD
    
    with np.errstate(divide='ignore'):
        mRR = mRRs / Th2
    
    S11 = dRR
    
    S12 = np.empty(len(dRR))
    for i_s12 in range(len(dRR)):
        if i_s12 == 0:
            lim_min = 0
        else:
            lim_min = i_s12 - 1
        if i_s12 == len(dRR):
            lim_max = len(dRR)
        else:
            lim_max = i_s12 + 1
        drr_array = dRR[lim_min:lim_max]
        if dRR[i_s12] > 0:
            S12[i_s12] = np.max(drr_array)
        else:
            S12[i_s12] = np.min(drr_array)
    
    
    S21 = dRR
    
    S22 = np.empty(len(dRR))
    for i_s22 in range(len(dRR)-1):
        if i_s22 > len(dRR)-2:
            lim_max = len(dRR)
        else:
            lim_max = i_s22 + 2
        drr_array = dRR[i_s22+1:lim_max]
        if dRR[i_s22] >= 0:
            S22[i_s22] = np.max(drr_array)
        else:
            S22[i_s22] = np.min(drr_array)
    
    
    
    inds_st = np.where(np.abs(dRR)>1)
    
    ## Find ectopic beats
    c1 = 0.13
    c2 = 0.17
    
    ec1_ind1 = np.where(S11>1)
    S11_ = S11*(-1*c1)+c2
    ec1_ind2 = np.where(S12<S11_)
    u, c = np.unique(np.append(ec1_ind1, ec1_ind2), return_counts=True)
    ec1_ind = u[c > 1]
    
    ec2_ind1 = np.where(S11<-1)
    S11_ = S11*(-1*c1)-c2
    ec2_ind2 = np.where(S12>S11_)
    u, c = np.unique(np.append(ec2_ind1, ec2_ind2), return_counts=True)
    ec2_ind = u[c > 1]
    
    #### eq1 and eq2
    ec_ind = np.append(ec1_ind, ec2_ind)
    
    
    ## Find long or short beats
    
    inds_s21 = np.where(np.abs(dRR)>1)
    #### eq4
    inds_s22 = np.where(np.abs(mRR)>3)
    
    inds_s2_all = np.unique(np.append(inds_s21, inds_s22))
    
    dRR_mod = np.append(dRR, np.array([dRR[-1],dRR[-1]]))
    #### eq3
    ls1_ind1 = inds_s2_all[np.where(np.sign(dRR[inds_s2_all])*dRR_mod[inds_s2_all+1]<-1)]
    #### eq5
    ls1_ind2 = inds_s2_all[np.where(np.sign(dRR[inds_s2_all])*dRR_mod[inds_s2_all+2]<-1)]

    
    inds_s3_all = np.unique(np.append(ls1_ind1, ls1_ind2))
    
    #### eq6
    with np.errstate(invalid='ignore'):
        inds_s31 = inds_s3_all[np.where(np.abs(hr_c[inds_s3_all]/2-medRR[inds_s3_all])<Th2[inds_s3_all])]
    
    
    #### eq7
    # inds_s31 = inds_s3_all[np.where(np.abs((hr_c[inds_s3_all]+hr_c[inds_s3_all+1])-medRR[inds_s3_all])<Th2[inds_s3_all])]
    
    ## Ectopic beats 
    ectopics = ec_ind # 'x','r'
    
    ## Missed beats
    missed = inds_s31 # 'x','b'
    
    ## Extra beats
    extra = np.unique(np.append(inds_s22, ls1_ind1))  # 'x','g'
    
    ## Long/short 1 beats
    longshort1 = inds_s31  # '+','k'
    
    ## Long/short 2 beats
    longshort2 = ls1_ind2  # 'x','k'
    
    
    all_bad_peaks = np.unique(np.concatenate((ectopics, missed, extra, longshort1, longshort2)))


    return all_bad_peaks