import numpy as np
# import argparse
# import platform
# import math
# from torch import nn
# from torch.nn import functional as F
# from torch import optim
# import random
# from collections import deque, namedtuple
# import time
# from datetime import datetime
# import gym
# import torch
# import csv
# import pickle
# import plotly
# import plotly.graph_objs as go
# from torch import multiprocessing as mp
import sys
import os
# sys.path.append(/home/mainul/DQN/lib_dvh/)
sys.path.append('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/')
# from score_calcu import planIQ
# from TP_DVH_algo1 import runOpt_dvh
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

from math import sqrt
# from torch.distributions import Categorical
import numpy as np
# from gym import spaces
# import matplotlib.pyplot as plt
import random
# # from sklearn.model_selection import train_test_split

# ##################################################
# from collections import deque
# # import dqn_dvh_external_network
# import h5sparse
# import h5py
# from scipy.sparse import vstack
# from typing import List
# from scipy.sparse import csr_matrix
import numpy.linalg as LA
# import time
#################################################

# import logging
# # The next line is for importing from CORT data
# from data_prep_parth_complete_onceagain import loadDoseMatrix,loadMask,ProcessDmat
# the next line is for importing new UTSW testing data
from lib_dvh.data_prep import loadDoseMatrix,loadMask,ProcessDmat
# the next line is for importing the TORTS data
# from Prostate_TORTS.data_prep_TORTS_to_call import loadDoseMatrix,loadMask,ProcessDmat
# from lib_dvh.myconfig import *
pdose = 1 # target dose for PTV
maxiter = 40 # maximum iteration number for treatment planing optimization
##################################### ActorCritic Network ###############################

# from gym import Env
# from gym.spaces import Discrete
INPUT_SIZE = 100

from math import pi
def planIQ_train(MPTV, MBLA, MREC, xVec,pdose,check):
    # score of treatment plan, two kinds of scores:
    # 1: score from standard criterion, 2: score_fined for self-defined in order to emphasize ptv
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    scoreall = np.zeros((11,))
    #    tt = time.time()
    max_limit = 1

    # This is the code that I received from parvat
    # ind = round(0.03 / 0.015) - 1


    # # avg_DPTV = (DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]+DPTV[ind + 2] + DPTV[ind + 3]) / 5
    # avg_DPTV = (DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]) / 3
    # # avg_DPTV = DPTV[ind]
    # # if check == True:
    # #     print("avg_DPTV:",avg_DPTV)
    # score2 =  (avg_DPTV - 1.1)/(-0.03)
    # if score2 > max_limit:
    #     score2 = 1
    # if score2 < 0:
    #     score2 = 0
    # delta2 = 0.08
    # if (avg_DPTV > 1.05):
    #     score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    # else:
    #     score2_fine = 6########################################
    # # score2_fine = score2
    # scoreall[0] = score2

    # This is the 1st new one I am employing
    # size of a voxel = 0.027 cc
    # line 1
    avg_DPTV = DPTV[0]
    # line 2
    # avg_DPTV = (DPTV[0]+DPTV[1])/(54/30)
    score2 = avg_DPTV - 1.1
    if score2 >= 0:
        score2 = 0
    if score2 < 0:
        score2 = 1 
    delta2 = 0.08
    if (avg_DPTV > 1.05):
        score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    else:
        score2_fine = 6########################################
    # score2_fine = score2
    scoreall[0] = avg_DPTV


    DBLA1 = DBLA[DBLA >= 1.01]
    avg_DBLA1 = DBLA1.shape[0] / DBLA.shape[0]
    score5 = (avg_DBLA1 - 0.2 )/(-0.05)
    if score5 > max_limit:
        score5 = 1
    if score5 < 0:
        score5 = 0
    delta3 = 0.05
    if avg_DBLA1 < 0.2:
        score3_fine = 1 / pi * np.arctan(-(avg_DBLA1 - 0.175) / delta3) + 0.5
    else:
        score3_fine = 0
    scoreall[3] = score5

    DBLA2 = DBLA[DBLA >= pdose * 0.947]
    avg_DBLA2 = DBLA2.shape[0] / DBLA.shape[0]
    score6 = (avg_DBLA2 - 0.3 )/(-0.05)
    if score6 > max_limit:
        score6 = 1
    if score6 < 0:
        score6 = 0
    delta4 = 0.05
    if avg_DBLA2 < 0.3:
        score4_fine = 1 / pi * np.arctan(-(avg_DBLA2 - 0.55 / 2) / delta4) + 0.5
    else:
        score4_fine = 0
    scoreall[4] = score6

    DBLA3 = DBLA[DBLA >= 0.8838]
    avg_DBLA3 = DBLA3.shape[0] / DBLA.shape[0]
    score7 = (avg_DBLA3 - 0.4 )/(-0.05)
    if score7 > max_limit:
        score7 = 1
    if score7 < 0:
        score7 = 0
    delta5 = 0.05
    if avg_DBLA3 < 0.4:
        score5_fine = 1 / pi * np.arctan(-(avg_DBLA3 - 0.75 / 2) / delta5) + 0.5
    else:
        score5_fine = 0
    scoreall[5] = score7

    DBLA4 = DBLA[DBLA >= 0.8207]
    avg_DBLA4 = DBLA4.shape[0] / DBLA.shape[0]
    score8 = (avg_DBLA4 - 0.55)/(-0.05)
    if score8 > max_limit:
        score8 = 1
    if score8 < 0:
        score8 = 0
    delta6 = 0.05
    if avg_DBLA4 < 0.55:
        score6_fine = 1 / pi * np.arctan(-(avg_DBLA4 - 1.05 / 2) / delta6) + 0.5
    else:
        score6_fine = 0
    scoreall[6] = score8

    DREC1 = DREC[DREC >= 0.947]
    avg_DREC1 = DREC1.shape[0] / DREC.shape[0]
    score9 = (avg_DREC1 - 0.2)/(-0.05)
    if score9 > max_limit:
        score9 = 1
    if score9 < 0:
        score9 = 0
    delta7 = 0.05
    if avg_DREC1 < 0.2:
        score7_fine = 1 / pi * np.arctan(-(avg_DREC1 - 0.35 / 2) / delta7) + 0.5
    else:
        score7_fine = 0
    scoreall[7] = score9

    DREC2 = DREC[DREC >= 0.8838]
    avg_DREC2 = DREC2.shape[0] / DREC.shape[0]
    score10 = (avg_DREC2 - 0.3)/(-0.05)
    if score10 > max_limit:
        score10 = 1
    if score10 < 0:
        score10 = 0
    delta8 = 0.05
    if avg_DREC2 < 0.3:
        score8_fine = 1 / pi * np.arctan(-(avg_DREC2 - 0.55 / 2) / delta8) + 0.5
    else:
        score8_fine = 0

    scoreall[8] = score10

    DREC3 = DREC[DREC >= 0.8207]
    avg_DREC3 = DREC3.shape[0] / DREC.shape[0]
    score11 = (avg_DREC3 - 0.4)/(-0.05)
    if score11 > max_limit:
        score11 = 1
    if score11 < 0:
        score11 = 0
    delta9 = 0.05
    if avg_DREC3 < 0.4:
        score9_fine = 1 / pi * np.arctan(-(avg_DREC3 - 0.75 / 2) / delta9) + 0.5
    else:
        score9_fine = 0

    scoreall[9] = score11

    DREC4 = DREC[DREC >= 0.7576]
    avg_DREC4 = DREC4.shape[0] / DREC.shape[0]
    score12 = (avg_DREC4 - 0.55)/(-0.05)
    if score12 > max_limit:
        score12 = 1
    if score12 < 0:
        score12 = 0
    delta10 = 0.05
    if avg_DREC4 < 0.55:
        score10_fine = 1 / pi * np.arctan(-(avg_DREC4 - 1.05 / 2) / delta10) + 0.5
    else:
        score10_fine = 0

    scoreall[10] = score12
    #    elapsedTime = time.time()-tt
    #    print('time:{}',format(elapsedTime))

    score = score2 + score5 + score6 + score7 + score8 + score9 + score10 + score11 + score12
    if check == True:
        print(score2, score5, score6, score7, score8, score9, score10, score11, score12)
    if score2_fine > 0.5:
        score_fine = score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine
    else:
        score_fine = (
                    score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine)

    return score_fine, score, scoreall


def MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec, gamma,pdose,maxiter):
    # treatment planning optimization in DVH-based scheme
    beta=2
    lambdaBLA = lambdaBLA/lambdaPTV
    lambdaREC = lambdaREC/lambdaPTV
    # xVec = np.ones((MPTV.shape[1],))
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
#    MPTV95 = MPTV[DPTV>=D95,:]
    factor = pdose / D95
    xVec = xVec * factor
    y = MPTV.dot(xVec)
    MPTVT = MPTV.transpose()
    DPTV = MPTV.dot(xVec)
    for iter in range(maxiter):
        xVec_old = xVec
        DPTV1 = np.sort(DPTV)
        posi = int(round((1 - VPTV) * DPTV1.shape[0]))-1
        if posi < 0:###### what was missing
            posi = 0###### what was missing
        DPTVV = DPTV1[posi]

        # posi = int(round(0.05 * DPTV1.shape[0]))-1
        # D95 = DPTV1[posi]
        # MPTV95 = MPTV[DPTV >= D95, :]
        # DPTV95 = MPTV95.dot(xVec)
        DBLA = MBLA.dot(xVec)
        DBLA1 = np.sort(DBLA)
        posi = int(round((1 - VBLA) * DBLA1.shape[0]))-1
        if posi < 0:
            posi = 0
        DBLAV = DBLA1[posi]
        DREC = MREC.dot(xVec)
        DREC1 = np.sort(DREC)
        posi = int(round((1 - VREC) * DREC1.shape[0]))-1
        if posi < 0:
            posi = 0
        DRECV = DREC1[posi]

        MPTVV =  MPTV[DPTV>=DPTVV,:]
        temp= DPTV[DPTV>=DPTVV]
        if np.max(temp) > pdose* tPTV:
            MPTV1 = MPTVV[temp > pdose*tPTV, :]
            targetPTV1 = pdose*tPTV*np.ones((MPTV1.shape[0],))
            MPTV1T = MPTV1.transpose()
            temp1 = MPTV1.dot(xVec)
            temp1 = MPTV1T.dot(temp1)
            temp1 = temp1 * 1/MPTV1.shape[0]
            b1 = MPTV1T.dot(targetPTV1) / MPTV1.shape[0]
        else:
            temp1 = np.zeros((xVec.shape))
            b1 = np.zeros((xVec.shape))
            tempp1 = np.zeros((xVec.shape))
        tempptv = temp

        temp2 = MPTV.dot(xVec)
        temp2 = beta*MPTVT.dot(temp2)/y.shape[0]
        b2 =  beta*MPTVT.dot(y)/y.shape[0]


        MBLAV = MBLA[DBLA>=DBLAV,:]
        temp = DBLA[DBLA>=DBLAV]
        if np.max(temp) > pdose * tBLA:
            MBLA1 = MBLAV[temp > pdose * tBLA, :]
            targetBLA1 = pdose*tBLA*np.ones((MBLA1.shape[0],))
            MBLA1T = MBLA1.transpose()
            temp3 = MBLA1.dot(xVec)
            temp3 = MBLA1T.dot(temp3)
            temp3 = temp3 * lambdaBLA/MBLA1.shape[0]
            b3 = lambdaBLA * MBLA1T.dot(targetBLA1) / max(MBLA1.shape[0], 1)
        else:
            temp3 = np.zeros((xVec.shape))
            b3 = np.zeros((xVec.shape))
            tempp3 = np.zeros((xVec.shape))
        tempbla = temp

        MRECV = MREC[DREC >= DRECV, :]
        temp = DREC[DREC >= DRECV]
        if np.max(temp) > pdose * tREC:
            MREC1 = MRECV[temp > pdose * tREC, :]
            targetREC1 = pdose*tREC*np.ones((MREC1.shape[0],))
            MREC1T = MREC1.transpose()
            temp4 = MREC1.dot(xVec)
            temp4 = MREC1T.dot(temp4)
            temp4 = temp4 * lambdaREC/MREC1.shape[0]
            b4 = lambdaREC * MREC1T.dot(targetREC1) / MREC1.shape[0]
        else:
            temp4 = np.zeros((xVec.shape))
            b4 = np.zeros((xVec.shape))
            tempp4 = np.zeros((xVec.shape))
        temprec = temp

        templhs = temp1+temp2+temp3+temp4
        b = b1+b2+b3+b4-MPTVT.dot(gamma)
        r = b - templhs
        p = r
        rsold = np.inner(r,r)

        # print("rsold=", rsold, "iter=", iter, "=========================")  # this not

        if rsold>1e-10:
            for i in range(3):
                if np.max(tempptv) > pdose*tPTV :
                    tempp1 = MPTV1.dot(p)
                    tempp1 = MPTV1T.dot(tempp1)
                    tempp1 = tempp1 * 1 / MPTV1.shape[0]


                tempp2 = MPTV.dot(p)
                tempp2 = beta * MPTVT.dot(tempp2)/y.shape[0]

                if np.max(tempbla) > pdose * tBLA:
                    tempp3 = MBLA1.dot(p)
                    tempp3 = MBLA1T.dot(tempp3)
                    tempp3 = tempp3 * lambdaBLA / MBLA1.shape[0]

                if np.max(temprec) > pdose * tREC:
                    tempp4 = MREC1.dot(p)
                    tempp4 = MREC1T.dot(tempp4)
                    tempp4 = tempp4 * lambdaREC / MREC1.shape[0]


                Ap = tempp1 + tempp2 + tempp3 + tempp4
                pAp = np.inner(p, Ap)
                alpha = rsold / pAp
                xVec = xVec + alpha * p
                xVec[xVec<0]=0
                r = r - alpha * Ap
                rsnew = np.inner(r, r)
                if np.sqrt(rsnew) < 1e-5:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew
        DPTV = MPTV.dot(xVec)
        y = (DPTV * beta/y.shape[0] + gamma) / (beta/y.shape[0])
        Dy = np.sort(y)
        posi = int(round(0.05 * Dy.shape[0]))
        D95 = Dy[posi]
        temp = np.zeros(y.shape)
        temp[y>=D95] = y[y>=D95]
        temp[temp<pdose] = pdose
        y[y>=D95] = temp[y>=D95]
        gamma = gamma + beta * (MPTV.dot(xVec)-y)/y.shape[0]

        if LA.norm(xVec - xVec_old, 2) / LA.norm(xVec_old, 2) < 5e-3:
            break
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
    factor = pdose / D95 # thresholidng
    xVec = xVec * factor
    converge = 1
    if iter == maxiter - 1:
        converge = 0
    # print("LOOKED HERE DAMON:",converge,iter)
    return xVec, iter


def runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter):
    # run optimization and generate DVH curves
    xVec, iter = MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC,lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
#    j = 0
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)


    ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((INPUT_SIZE + 1,))
    edge_ptv[1:INPUT_SIZE + 1] = np.linspace(pdose, pdose * 1.15, INPUT_SIZE)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

    edge_bladder = np.zeros((INPUT_SIZE + 1,))
    edge_bladder[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)
    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

    edge_rectum = np.zeros((INPUT_SIZE + 1,))
    edge_rectum[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)
    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

    Y = np.zeros((INPUT_SIZE, 3))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    Y = np.reshape(Y, (100 * 3,), order='F')




    return Y, iter, xVec



def runOpt_Fulldvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter):
    # run optimization and generate DVH curves
    xVec, iter = MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC,lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
#    j = 0
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)


    ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((100 + 1,))
    edge_ptv_max = np.zeros((100 + 1,))                        
    edge_ptv[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
    edge_ptv_max[1:100 + 1] = np.linspace(0, max(DPTV), 100)
    x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    (n_ptv_max, b_max) = np.histogram(DPTV, bins=edge_ptv_max)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
    y_ptv_max = 1 - np.cumsum(n_ptv_max / len(DPTV), axis=0)

    edge_bladder = np.zeros((100 + 1,))
    edge_bladder_max = np.zeros((100 + 1,))
    edge_bladder[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
    edge_bladder_max[1:100 + 1] = np.linspace(0, max(DBLA), 100)                            
    x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    (n_bladder_max, b_max) = np.histogram(DBLA, bins = edge_bladder_max)
    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
    y_bladder_max = 1 - np.cumsum(n_bladder_max/len(DBLA), axis = 0)

    edge_rectum = np.zeros((100 + 1,))
    edge_rectum_max = np.zeros((100 + 1,))
    edge_rectum[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
    edge_rectum_max[1:100 + 1] = np.linspace(0, max(DREC), 100)                            
    x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    (n_rectum_max, b_max) = np.histogram(DREC , bins = edge_rectum_max)
    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
    y_rectum_max = 1 - np.cumsum(n_rectum_max / len(DREC), axis = 0)

    Y = np.zeros((100, 12))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    # X = np.zeros((1000, 3))
    Y[:, 3] = x_ptv
    Y[:, 4] = x_bladder
    Y[:, 5] = x_rectum

    # storing max range histograms
    Y[:, 6] = y_ptv_max
    Y[:, 7] = y_bladder_max
    Y[:, 8] = y_rectum_max

    Y[:, 9] = edge_ptv_max[1:100+1]
    Y[:, 10] = edge_bladder_max[1:100+1]
    Y[:, 11] = edge_rectum_max[1:100+1]
    
    # Y = np.reshape(Y, (100 * 3,), order='F')




    return Y

import math as m

# repList = ['010']
#repList = ['017']
repList = ['010']

data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'
data_path2 = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'


# The following block is for testing all UTSW dataset
for i in range(len(repList)):
    print("len(repList)", len(repList))
    print('patient No', repList[i])
    globals()['doseMatrix_' + repList[i]] = loadDoseMatrix(data_path + repList[i] + '.hdf5')
    print("doseMatrix loaded")
    globals()['targetLabels_' + repList[i]], globals()['bladderLabel' + repList[i]], globals()[
      'rectumLabel' + repList[i]], \
    globals()['PTVLabel' + repList[i]] = loadMask(data_path2 + repList[i] + '.h5')
    print("PTVLabel loaded")
    print(globals()['doseMatrix_' + repList[i]].shape)


sampleid = repList[0]
doseMatrix = globals()['doseMatrix_' + str(sampleid)]
targetLabels = globals()['targetLabels_' + str(sampleid)]
bladderLabel = globals()['bladderLabel' + str(sampleid)]
rectumLabel = globals()['rectumLabel' + str(sampleid)]
PTVLabel = globals()['PTVLabel' + str(sampleid)]

MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel,
                                                                 rectumLabel)

TPPs = np.load("/data2/mainul/DQNFGSMUTSWallPaper/020tpptuning0.npz")
print(TPPs)
for key in TPPs.keys():
    print(f"Array '{key}':")
    print(TPPs[key])



tPTV = 1 
tBLA = 1
tREC = 1
lambdaPTV = 1
lambdaBLA = 1
lambdaREC = 1
VPTV = 0.1
VBLA = 1
VREC = 1
xVec = np.ones((MPTV.shape[1],))
gamma = np.zeros((MPTV.shape[0],))
# lambdaBLA = lambdaBLA*m.exp(0.5)
print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
    tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)
# --------------------- solve treatment planning optmization -----------------------------
state_test0 = \
    runOpt_Fulldvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC,
               lambdaPTV, lambdaBLA, lambdaREC,
               VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)


print(state_test0)

Y = state_test0

y_ptv = Y[:, 6]
y_bladder = Y[:, 7]
y_rectum = Y[:, 8]

y_ptv = y_ptv*100
y_bladder = y_bladder*100
y_rectum = y_rectum*100

# y_ptv = y_ptv/max(y_ptv)
# y_bladder = y_bladder/max(y_bladder)
# y_rectum = y_rectum/max(y_rectum)

x_ptv = Y[:, 9]
x_bladder = Y[:, 10]
x_rectum = Y[:, 11]

x_ptv = x_ptv*100
x_bladder = x_bladder*100
x_rectum = x_rectum*100


# Create a plot
# plt.figure(figsize=(10, 6))

# Plot the three datasets against the array of points
plt.plot(x_ptv, y_ptv, label='PTV1_p', color='red', linestyle='-')
plt.plot(x_bladder, y_bladder, label='Bla1_p', color='green', linestyle='-' )
plt.plot(x_rectum, y_rectum, label='Rec1_p', color='blue', linestyle='-')



# the following for gettting the convolution of the whole image ==========================
sigma = 3.0
kernel_size = int(18 * sigma + 1)  # 6*sigma gives a wide enough kernel for most uses
x = np.linspace(-9*sigma, 9*sigma, kernel_size)

# 1D Gaussian kernel
gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel
# =========================================================================================


def gaussian(x, sigma, peak, mu):
    y = np.zeros(x.size)
    for i in range(x.size):
        gauss = peak*np.exp((-0.5 * ((x[i]-mu) / sigma) ** 2))
        y[i] = gauss

    return y

# The follwoing kernel is for getting the convolution for an individual point
# y_ptvIndBlurr = np.exp(0.5)
# y_ptvIndBlurr[22] = convolve1d(y_ptv[20:26], gaussian_kernel)[0]
y_bladderIndBlurr = np.arange(0,100)
y_bladderIndBlurr = gaussian(y_bladderIndBlurr, 0.7, y_bladder[22], y_bladderIndBlurr[22] )
# print('y_bladderIndBlurr', y_bladderIndBlurr)
# y_bladderIndBlurr[22] = y_bladder[20]
# y_bladderIndBlurr 
# y_rectumIndBlurr = np.zeros(100)
# y_rectumIndBlurr[22] = convolve1d(y_rectum[20:26], gaussian_kernel)[0]


# y_ptvpadding = np.zeros(3)
# y_rectumpadding = np.zeros(3)
# y_bladderpadding = np.zeros(3)

# y_ptvafterpadding = np.concatenate((y_ptv, y_ptvpadding), axis = 0)
# y_rectumafterpadding = np.concatenate((y_ptv, y_rectumpadding), axis = 0)
# y_bladderafterpadding = np.concatenate((y_ptv, y_bladderpadding), axis = 0)


# y_ptvBlurrExtra = convolve1d(y_ptvafterpadding, gaussian_kernel)
# y_rectumBlurrExtra = convolve1d(y_rectumafterpadding, gaussian_kernel)
# y_bladderBlurrExtra = convolve1d(y_bladderafterpadding, gaussian_kernel)


# y_ptvBlurr = convolve1d(y_ptv, gaussian_kernel, mode = 'constant', cval = 0)
# y_rectumBlurr = convolve1d(y_rectum, gaussian_kernel, mode = 'constant', cval = 0)
# y_bladderBlurr = convolve1d(y_bladder, gaussian_kernel, mode = 'constant', cval = 0)

y_ptvBlurr = convolve1d(y_ptv, gaussian_kernel)
y_rectumBlurr = convolve1d(y_rectum, gaussian_kernel)
y_bladderBlurr = convolve1d(y_bladder, gaussian_kernel)

# plt.plot(x_ptv, y_ptvBlurr, label='PTV1_p', color='red', linestyle='--')
# plt.plot(x_bladder, y_bladderBlurr, label='Bla1_p', color='green', linestyle='--' )
# plt.plot(x_rectum, y_rectumBlurr, label='Rec1_p', color='blue', linestyle='--')
y_bladderFinal = y_bladder*(1- y_bladderIndBlurr) + y_bladderBlurr*y_bladderIndBlurr


# plt.plot(x_ptv, y_ptvIndBlurr, label='PTV1_p', color='red', linestyle='--')
plt.plot(x_bladder, y_bladderFinal, label='Bla1_p', color='green', linestyle='--' )
# plt.plot(x_rectum, y_rectumIndBlurr, label='Rec1_p', color='blue', linestyle='--')

plt.xlim(0,120)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)

# Add titles and labels
# plt.title('Adversarial Perturbation', fontsize = 14)
plt.xlabel('Relative Dose (%)', fontsize = 30)
plt.ylabel('Relative Volume (%)', fontsize = 30)
plt.legend(loc='best', fontsize = 30)
plt.tight_layout()
# plt.savefig('/data2/mainul/ExplainableAIResults/convolvedDVHSigma1.png', dpi = 1200)
# plt.savefig('/data2/mainul/DataAndGraph/ScoreGettingbetterACERFull.png', dpi = 1200)
plt.show()

# The next part is unnecesssary and scheduled for deletion.

# np.save('/data2/mainul/DataAndGraphDQN/FGSMPaper/PaperRepresantativeUnperturbedFull', state_test0)
# planScore_fine, planScore,scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec,pdose, True)
# print(planScore)

# # DQN patient '010' before Perturbation ================================
# # tREC = tREC*0.8

# # DQN patient '010' after Perturbation ================================
# lambdaREC = lambdaREC*m.exp(0.5)

# # ACER patient '010' before and after Perturbation ================================
# tBLA = tBLA*0.6

# # lambdaBLA = lambdaBLA*m.exp(0.5)
# # lambdaBLA = lambdaBLA*m.exp(-0.5)
# # lambdaBLA = lambdaBLA*m.exp(-0.5)
# # if action == 0:
# #     tPTV = min(tPTV * 1.01, paraMax_tPTV)
# # elif action == 1:
# #     tPTV = max(tPTV * 0.91, paraMin_tPTV)
# # elif action == 2:
# #     tBLA = min(tBLA * 1.25, paraMax_tOAR)
# # elif action == 3:
# #     tBLA = tBLA * 0.6
# # elif action == 4:
# #     tREC = min(tREC * 1.25, paraMax_tOAR)
# # elif action == 5:
# #     tREC = tREC * 0.6
# # elif action == 6:
# #     lambdaPTV = lambdaPTV * 1.65
# # elif action == 7:
# #     lambdaPTV = lambdaPTV * 0.6
# # elif action == 8:
# #     lambdaBLA = lambdaBLA * 1.65
# # elif action == 9:
# #     lambdaBLA = lambdaBLA * 0.6
# # elif action == 10:
# #     lambdaREC = lambdaREC * 1.65
# # elif action == 11:
# #     lambdaREC = lambdaREC * 0.6
# # elif action == 12:
# #     VPTV = min(VPTV * 1.25, paraMax_VPTV)
# # elif action == 13:
# #     VPTV = VPTV * 0.8
# # elif action == 14:
# #     VBLA = min(VBLA * 1.25, paraMax_VOAR)
# # elif action == 15:
# #     VBLA = VBLA * 0.8
# # elif action == 16:
# #     VREC = min(VREC * 1.25, paraMax_VOAR)
# # elif action == 17:
# #     VREC = VREC * 0.8
# print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
#     tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)
# # --------------------- solve treatment planning optmization -----------------------------
# xVec = np.ones((MPTV.shape[1],))
# gamma = np.zeros((MPTV.shape[0],))

# state_test0 = \
#     runOpt_Fulldvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC,
#                lambdaPTV, lambdaBLA, lambdaREC,
#                VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)

# # For before perturbation ====================================
# # np.save('/data2/mainul/DataAndGraphDQN/FGSMPaper/PaperRepresantativeBeforeperturbedDQNFull', state_test0)
# # np.save('/data2/mainul/DataAndGraph/PaperRepresantativeBeforeperturbedACERFull', state_test0)

# # For after Perturbation =====================================
# np.save('/data2/mainul/DataAndGraph/PaperRepresantativeAfterperturbedACERFull', state_test0)
# # np.save('/data2/mainul/DataAndGraphDQN/FGSMPaper/PaperRepresantativeAfterperturbedDQNFull', state_test0)

# planScore_fine, planScore,scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec,pdose, True)
# print(planScore)
