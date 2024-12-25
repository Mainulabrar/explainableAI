# -*- coding: utf-8 -*-
import argparse
import platform
import math
from torch import nn
from torch.nn import functional as F
from torch import optim
import random
from collections import deque, namedtuple
import time
from datetime import datetime
import gym
import torch
import csv
import pickle
import plotly
import plotly.graph_objs as go
from torch import multiprocessing as mp
import os
# import CV
import sys
sys.path.append('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/')
from scipy.ndimage import convolve1d


from math import sqrt
from torch.distributions import Categorical
import numpy as np
from gym import spaces
# import matplotlib.pyplot as plt
import random
# from sklearn.model_selection import train_test_split

##################################################
from collections import deque
# import dqn_dvh_external_network
import h5sparse
import h5py
from scipy.sparse import vstack
from typing import List
from scipy.sparse import csr_matrix
import numpy.linalg as LA
import time
#################################################

# import logging
# # The next line is for importing from CORT data
from data_prep_parth_complete_onceagain import loadDoseMatrix,loadMask,ProcessDmat
# the next line is for importing new UTSW testing data
# from lib_dvh.data_prep import loadDoseMatrix,loadMask,ProcessDmat
# the next line is for importing the TORTS data
# from Prostate_TORTS.data_prep_TORTS_to_call import loadDoseMatrix,loadMask,ProcessDmat
# from lib_dvh.myconfig import *
pdose = 1 # target dose for PTV
maxiter = 40 # maximum iteration number for treatment planing optimization
##################################### ActorCritic Network ###############################

from gym import Env
from gym.spaces import Discrete
INPUT_SIZE = 100  # DVH interval number
patient_list = ['01']

#######################################

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    # self.state_size = observation_space.shape[0]
    self.state_size = 300
    self.action_size = action_space.n


    self.fc1 = nn.Linear(self.state_size, hidden_size) # (2,32)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size) # (32,32)
    self.fc_actor = nn.Linear(hidden_size, self.action_size) # (32, 4)
    self.fc_critic = nn.Linear(hidden_size, self.action_size) # (32,4)

  def forward(self, x, h):

    # Here, x = state
    x = F.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h

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

    # This is the 2nd new one I am employing
    # size of a voxel = 0.027 cc
    # number of voxel = (184*184*90) = 3047040
    # Total Volume = (0.027*3047040) cc = 82270.08 cc
    # percentage of volume for 0.03 cc = (0.03/82270.08) 
    # # line 1
    # DPTV1 = DPTV[DPTV >= 1.1]
    # allowed_volume_percentage = 0.03/82270.08
    # # line 2
    # avg_DPTV = DPTV1.shape[0]/ DPTV.shape[0]
    # score2 = avg_DPTV - allowed_volume_percentage
    # if score2 >= 0:
    #     score2 = 0
    # if score2 < 0:
    #     score2 = 1 
    # delta2 = 0.08
    # if (avg_DPTV > 1.05):
    #     score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    # else:
    #     score2_fine = 6########################################
    # # score2_fine = score2
    # scoreall[0] = avg_DPTV

    # # This is the exact code Dr Chi suggests employing
    # # size of a voxel = 0.027 cc
    # avg_DPTV = (DPTV[0]*0.027 + DPTV[1]*0.003)/0.03
    # score2 = avg_DPTV - 1.1
    # if score2 >= 0:
    #     score2 = 0
    # if score2 < 0:
    #     score2 = 1 
    # delta2 = 0.08
    # if (avg_DPTV > 1.05):
    #     score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    # else:
    #     score2_fine = 6########################################
    # # score2_fine = score2
    # scoreall[0] = avg_DPTV


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

    # INPUT_SIZE = 100
    #
    # ## Plot DVH curve for optimized plan
    # edge_ptv = np.zeros((INPUT_SIZE+1,))
    # # edge_ptv[1:INPUT_SIZE+1] = np.linspace(pdose,pdose*1.15, INPUT_SIZE)
    # edge_ptv[1:INPUT_SIZE + 1] = np.linspace(0, max(DPTV), INPUT_SIZE)
    # # x_ptv = np.linspace(pdose+ 0.5* pdose*1.15/INPUT_SIZE,pdose*1.15,INPUT_SIZE)
    # x_ptv = np.linspace(0.5 * max(DPTV) / INPUT_SIZE, max(DPTV), INPUT_SIZE)
    # (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    # y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
    #
    # edge_bladder = np.zeros((INPUT_SIZE+1,))
    # # edge_bladder[1:INPUT_SIZE+1] = np.linspace(0.6*pdose, 1.1*pdose, INPUT_SIZE)
    # edge_bladder[1:INPUT_SIZE + 1] = np.linspace(0, max(DBLA), INPUT_SIZE)
    # # x_bladder = np.linspace(0.6*pdose+ 0.5* 1.1*pdose / INPUT_SIZE,1.1*pdose , INPUT_SIZE)
    # x_bladder = np.linspace(0.5 * max(DBLA) / INPUT_SIZE, max(DBLA), INPUT_SIZE)
    # (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    # y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
    #
    # edge_rectum = np.zeros((INPUT_SIZE+1,))
    # # edge_rectum[1:INPUT_SIZE+1] = np.linspace(0.6*pdose, 1.1*pdose, INPUT_SIZE)
    # edge_rectum[1:INPUT_SIZE + 1] = np.linspace(0, max(DREC), INPUT_SIZE)
    # # x_rectum = np.linspace(0.6*pdose+ 0.5* 1.1*pdose / INPUT_SIZE, 1.1*pdose, INPUT_SIZE)
    # x_rectum = np.linspace(0.5 * max(DREC) / INPUT_SIZE, max(DREC), INPUT_SIZE)
    # (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    # y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
    #
    # Y = np.zeros((INPUT_SIZE,3))
    # Y[:, 0] = y_ptv
    # Y[:, 1] = y_bladder
    # Y[:, 2] = y_rectum
    #
    # X = np.zeros((INPUT_SIZE, 3))
    # X[:, 0] = x_ptv
    # X[:, 1] = x_bladder
    # X[:, 2] = x_rectum
    #
    #
    #
    # # plt.plot(x_ptv, y_ptv)
    # # plt.plot(x_bladder, y_bladder)
    # # plt.plot(x_rectum, y_rectum)
    # # plt.legend(('ptv', 'bladder', 'rectum'))
    # # plt.title('Initial DVH')
    # # plt.savefig('Initial.png')
    # # plt.show(block=False)
    # # plt.close()
    # Y = np.reshape(Y,(100*3,),order = 'F')


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

import math as m
class TreatmentEnv(Env):
    """A Treatment planning environment for OpenAI gym"""

    # metedata = {'render.modes':['human']}
    # Set up the dimensions and type of space that the action and observation space is
    def __init__(self):
        self.action_space = Discrete(18)  # Box(low=np.array([0,0.5]), high=np.array([26,1.5]), dtype=np.float32)#Discrete (26)
        self.observation_space = np.zeros([300])

        # How many times it will loop per epoch
        self.time_limit = 30

    def step(self, action, t, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC,
             lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter):
        # The environment that the agent will work in includes both the Treatment planning system and the Reward function

        # Uncomment this part for action_original
        # xVec = np.ones((MPTV.shape[1],))
        # gamma = np.zeros((MPTV.shape[0],))
        # _, _, xVec, _ = \
        #     runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,
        #                gamma, pdose, maxiter)
        # state = np.reshape(state, [INPUT_SIZE * 3])
        # xVec = np.ones((MPTV.shape[1],))
        # Score_fine, Score, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
        self.action = action
        # print("action:", action)
        # Uncomment this part for original action and tab the code after else and before DPTV. Also, change actionnum in myconfig file
        # if action % 3 == 1:
        #     n_state = state
        #     # print('This is still the same')
        #     reward = 0
        #     action_factor = 1
        # else:
        paraMax = 100000  # change in validation as well
        paraMin = 0
        paraMax_tPTV = 1.2
        paraMin_tPTV = 1
        paraMax_tOAR = 1
        paraMax_VOAR = 1
        paraMax_VPTV = 0.3


        if action == 0:
            tPTV = min(tPTV * 1.01, paraMax_tPTV)
        elif action == 1:
            tPTV = max(tPTV * 0.91, paraMin_tPTV)
        elif action == 2:
            tBLA = min(tBLA * 1.25, paraMax_tOAR)
        elif action == 3:
            tBLA = tBLA * 0.6
        elif action == 4:
            tREC = min(tREC * 1.25, paraMax_tOAR)
        elif action == 5:
            tREC = tREC * 0.6
        elif action == 6:
            lambdaPTV = lambdaPTV * 1.65
        elif action == 7:
            lambdaPTV = lambdaPTV * 0.6
        elif action == 8:
            lambdaBLA = lambdaBLA * 1.65
        elif action == 9:
            lambdaBLA = lambdaBLA * 0.6
        elif action == 10:
            lambdaREC = lambdaREC * 1.65
        elif action == 11:
            lambdaREC = lambdaREC * 0.6
        elif action == 12:
            VPTV = min(VPTV * 1.25, paraMax_VPTV)
        elif action == 13:
            VPTV = VPTV * 0.8
        elif action == 14:
            VBLA = min(VBLA * 1.25, paraMax_VOAR)
        elif action == 15:
            VBLA = VBLA * 0.8
        elif action == 16:
            VREC = min(VREC * 1.25, paraMax_VOAR)
        elif action == 17:
            VREC = VREC * 0.8

        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        n_state, _, xVec = \
            runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,
                       gamma, pdose, maxiter)
        Score_fine1, Score1, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)

        extra = 0 if Score1 != 9 else 2

        # Uncomment this part original scoring system
        reward = (Score_fine1 - Score_fine) + (Score1 - Score) * 4
        Done = False
        if Score1 == 9:
            Done = True
        return n_state, reward, Score_fine1, Score1, Done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec

    def reset(self):
        self.state = self.observation_space
        self.time_limit = 30
        return self.state

    def close(self):
        pass

class TwoDGridWorld(gym.Env):
  # actions available
  UP = 0
  LEFT = 1
  DOWN = 2
  RIGHT = 3

  def __init__(self, size):
    super(TwoDGridWorld, self).__init__()

    ##############################################################################################################
    self.size = size  # size of the grid world
    self.agent_position = [0, 0]
    self.end_state = [3,3]
    self.action_space = spaces.Discrete(4)
    # self.positions = [[0,0], [3,3]]
    # self.positions = [[0,3], [3,0]]
    # self.positions = [[0,0], [3,0]]
    # self.positions = [[0,1], [3,1]]
    # self.positions = [[0,0], [0,3], [3,0], [3,3] ]
    # self.grid = [[0,0], [0,1], [0,2],[0,3], [1,0], [1,1], [1,2],[1,3], [2,0], [2,1], [2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
    # self.positions_train, self.positions_test = train_test_split(self.grid, test_size=0.20, random_state=45)
    # self.positions = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
    # self.positions = [[0,0], [0,1],[1,0], [1,1]]
    # self.positions = [[0,0], [0,1], [0,2],[0,3],[0,4], [1,0], [1,1], [1,2],[1,3],[1,4], [2,0], [2,1], [2,2],[2,3],[2,4],[3,0],[3,1],[3,2],[3,3],[3,4]]
    # self.positions = [[1,3], [3,1]]
    # set the observation space to (1,) to represent agent position in the grid world
    # staring from [0,size*size)
    # self.observation_space = spaces.Box(low=0, high=size, shape=(2,2,2), dtype=np.uint8)
    self.observation_shape = (size, size, 2)
    self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                        high=np.ones(self.observation_shape), dtype=np.uint8
                                        )

  def step(self, action):
    info = {}  # additional information
    reward = 0;

    finalx = self.end_state[0]
    finaly = self.end_state[1]
    currx = self.agent_position[0]
    curry = self.agent_position[1]
    if action == self.UP:
      newx = currx - 1
      newy = curry
      self.agent_position = [newx, newy]
      if not (newx > (self.size - 1) or newx < 0):
        # self.agent_position = [newx, newy]
        #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
        # if(abs(finalx - currx)+ abs(finaly-curry) < abs(finalx - newx)+abs(finaly-newy)):
        if (abs(finalx - currx) < abs(finalx - newx)):
          reward = -1
        if (finaly == newy and finalx == newx):
          reward = reward + 1
      else:
        reward = -1
    elif action == self.LEFT:
      newy = curry - 1
      newx = currx
      self.agent_position = [newx, newy]
      if not (newy > (self.size - 1) or newy < 0):
        # self.agent_position = [newx, newy]
        #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
        # if(abs(finaly - curry)+abs(finalx - currx) < abs(finaly - newy)+abs(finalx - newx)):
        if (abs(finaly - curry) < abs(finaly - newy)):
          reward = -1
        if (finaly == newy and finalx == newx):
          reward = reward + 1
      else:
        reward = -1
    elif action == self.DOWN:
      newx = currx + 1
      newy = curry
      self.agent_position = [newx, newy]
      if not (newx > (self.size - 1) or newx < 0):
        # self.agent_position = [newx, newy]
        #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
        # if(abs(finalx - currx)+ abs(finaly-curry) < abs(finalx - newx)+abs(finaly-newy)):
        if (abs(finalx - currx) < abs(finalx - newx)):
          reward = -1
        if (finaly == newy and finalx == newx):
          reward = reward + 1
      else:
        reward = -1
    elif action == self.RIGHT:
      newy = curry + 1
      newx = currx
      self.agent_position = [newx, newy]
      if not (newy > (self.size - 1) or newy < 0):
        # self.agent_position = [newx, newy]
        #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
        # if(abs(finaly - curry)+abs(finalx - currx) < abs(finaly - newy)+abs(finalx - newx)):
        if (abs(finaly - curry) < abs(finaly - newy)):
          reward = -1
        if (finaly == newy and finalx == newx):
          reward = reward + 1
      else:
        reward = -1
    done = bool(self.agent_position[0] == self.end_state[0] and self.agent_position[1] == self.end_state[1])
    # if done:
    #   print("done")
    # print("self.agent_position", self.agent_position)
    # print("numpy agent position",np.array(self.agent_position).astype(np.uint8) )
    # return np.array([self.agent_position]).astype(np.uint8), reward, done, info
    return np.array(self.agent_position).astype(np.uint8), reward, done, info

  # def step(self, action):
  #     info = {}  # additional information
  #     reward = 0;
  #
  #     finalx = self.end_state[0]
  #     finaly = self.end_state[1]
  #     currx = self.agent_position[0]
  #     curry = self.agent_position[1]
  #     if action == self.UP:
  #         newx = currx - 1
  #         newy = curry
  #         self.agent_position = [newx, newy]
  #         if not (newx > (self.size - 1) or newx < 0):
  #             # self.agent_position = [newx, newy]
  #             #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
  #             # if(abs(finalx - currx)+ abs(finaly-curry) < abs(finalx - newx)+abs(finaly-newy)):
  #             if ((newx == 0 and (newy == 1 or newy == 2)) or (newx == 1 and newy == 1) or (newx == 2 and (newy == 1 or newy == 3))):
  #                 reward = -200
  #             # if (abs(finalx - currx) < abs(finalx - newx)):
  #             #     reward = -1
  #
  #
  #             if (finaly == newy and finalx == newx):
  #                 reward = reward + 5
  #         else:
  #             reward = -200
  #     elif action == self.LEFT:
  #         newy = curry - 1
  #         newx = currx
  #         self.agent_position = [newx, newy]
  #         if not (newy > (self.size - 1) or newy < 0):
  #             # self.agent_position = [newx, newy]
  #             #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
  #             # if(abs(finaly - curry)+abs(finalx - currx) < abs(finaly - newy)+abs(finalx - newx)):
  #             if ((newx == 0 and (newy == 1 or newy == 2)) or (newx == 1 and newy == 1) or (newx == 2 and (newy == 1 or newy == 3))):
  #                 reward = -200
  #             # if (abs(finaly - curry) < abs(finaly - newy)):
  #             #     reward = -1
  #
  #
  #             if (finaly == newy and finalx == newx):
  #                 reward = reward + 5
  #         else:
  #             reward = -200
  #     elif action == self.DOWN:
  #         newx = currx + 1
  #         newy = curry
  #         self.agent_position = [newx, newy]
  #         if not (newx > (self.size - 1) or newx < 0):
  #             # self.agent_position = [newx, newy]
  #             #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
  #             # if(abs(finalx - currx)+ abs(finaly-curry) < abs(finalx - newx)+abs(finaly-newy)):
  #             if ((newx == 0 and (newy == 1 or newy == 2)) or (newx == 1 and newy == 1) or (newx == 2 and (newy == 1 or newy == 3))):
  #                 reward = -200
  #             # if (abs(finalx - currx) < abs(finalx - newx)):
  #             #     reward = -1
  #
  #
  #             if (finaly == newy and finalx == newx):
  #                 reward = reward + 5
  #         else:
  #             reward = -200
  #     elif action == self.RIGHT:
  #         newy = curry + 1
  #         newx = currx
  #         self.agent_position = [newx, newy]
  #         if not (newy > (self.size - 1) or newy < 0):
  #             # self.agent_position = [newx, newy]
  #             #             if(sqrt((finalx - currx)**2+ (finaly-curry)**2) < sqrt((finalx - newx)**2+(finaly-newy)**2)):
  #             # if(abs(finaly - curry)+abs(finalx - currx) < abs(finaly - newy)+abs(finalx - newx)):
  #             if ((newx == 0 and (newy == 1 or newy == 2)) or (newx == 1 and newy == 1) or (newx == 2 and (newy == 1 or newy == 3))):
  #                 reward = -200
  #             # if (abs(finaly - curry) < abs(finaly - newy)):
  #             #     reward = -1
  #
  #
  #             if (finaly == newy and finalx == newx):
  #                 reward = reward + 5
  #         else:
  #             reward = -200
  #
  #     # #Trying to find the shortest path
  #     # reward = reward - 1
  #
  #
  #     # if reward < -1:
  #     #     reward = reward - 10
  #
  #     done = bool(self.agent_position[0] == self.end_state[0] and self.agent_position[1] == self.end_state[1])
  #     # if done:
  #     #   print("done")
  #     # print("self.agent_position", self.agent_position)
  #     # print("numpy agent position",np.array(self.agent_position).astype(np.uint8) )
  #     # return np.array([self.agent_position]).astype(np.uint8), reward, done, info
  #     return np.array(self.agent_position).astype(np.uint8), reward, done, info


  def render(self):
    '''
        render the state
    '''

    if True:
      # print("n_games", n_games)
      print("agent_position",self.agent_position[0],self.agent_position[1])
      for r in range(self.size):
        for c in range(self.size):
          if r == self.agent_position[0] and c == self.agent_position[1]:
            print("X", end='')
          else:
            print('.', end='')
        print('')
      print("================================================================\n")

  def reset(self):

    # Uncomment this block of code if you want to train for the robot to go from one end to another end.
    self.agent_position = [0, 0]
    self.end_state = [3,3]
    # return np.array([self.agent_position]).astype(np.uint8)
    return np.array(self.agent_position).astype(np.uint8)
    #######################################################
    # self.agent_position = random.choice(self.positions_train)
    # self.end_state = [2,2]
    # while (self.agent_position[0] == self.end_state[0] and self.agent_position[1] == self.end_state[1]):
    #     self.agent_position = random.choice(self.positions_train)
    # self.agent_position = [self.agent_position[0],self.agent_position[1]]
    # print("", self.agent_position)
    # return np.array([self.agent_position]).astype(np.uint8)

  def reset_testing(self):
    # Uncomment this block of code if you want to test for the robot to go from one end to another end.
    self.agent_position = [0, 0]
    self.end_state = [3, 3]
    return np.array([self.agent_position]).astype(np.uint8)
    #######################################################
    # print("self.positions_train",self.positions_train)
    # print("self.positions_test",self.positions_test)
    # self.agent_position = random.choice(self.positions_test)
    # self.end_state = [2,2]
    # while (self.agent_position[0] == self.end_state[0] and self.agent_position[1] == self.end_state[1]):
    #     self.agent_position = random.choice(self.positions_test)
    # self.agent_position = [self.agent_position[0],self.agent_position[1]]
    # print("", self.agent_position)
    # return np.array([self.agent_position]).astype(np.uint8)

  def close(self):
    pass

class SharedRMSprop(optim.RMSprop):
  def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
    super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

    # State initialisation (must be done before step, else will not be shared between threads)
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = p.data.new().resize_(1).zero_()
        state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'].share_memory_()
        state['square_avg'].share_memory_()

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self.state[p]

        square_avg = state['square_avg']
        alpha = group['alpha']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], p.data)

        # g = αg + (1 - α)Δθ^2
        # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        # θ ← θ - ηΔθ/√(g + ε)
        avg = square_avg.sqrt().add_(group['eps'])
        # p.data.addcdiv_(-group['lr'], grad, avg)
        p.data.addcdiv_(grad, avg, value=-group['lr'])

    return loss

# from train import train########################################################


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))


class EpisodicReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.trajectory = []

  def append(self, state, action, reward, policy):
    self.trajectory.append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append(self.trajectory)
      self.trajectory = []
  # Samples random trajectory
  def sample(self, maxlen=0):
    mem = self.memory[random.randrange(len(self.memory))]
    T = len(mem)
    # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
    if maxlen > 0 and T > maxlen + 1:
      t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
      return mem[t:t + maxlen + 1]
    else:
      return mem

  # Samples batch of trajectories, truncating them to the same length
  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    minimum_size = min(len(trajectory) for trajectory in batch)
    batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def length(self):
    # Return number of epsiodes saved in memory
    return len(self.memory)

  def __len__(self):
    return sum(len(episode) for episode in self.memory)


# Converts a state from the OpenAI Gym (a numpy array) to a batch tensor
def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return max(k - 1, 0)


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr


# Updates networks
def _update_networks(args, T, model, shared_model, shared_average_model, loss, optimiser):
  # Zero shared and local grads
  # print("optimizer", optimiser)
  # print("parameters before zero grad started")
  # for param in model.parameters():
  #   print(param.grad)
  # print("parameters before zero grad ended")
  optimiser.zero_grad()
  # print("parameters after zero grad started")
  # for param in model.parameters():
  #   print(param.grad)
  # print("parameters after zero grad ended")
  """
  Calculate gradients for gradient descent on loss functions
  Note that math comments follow the paper, which is formulated for gradient ascent
  """
  # print("loss",loss)
  loss.backward()
  # Gradient L2 normalisation
  nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

  # Update shared_average_model
  for shared_param, shared_average_param in zip(shared_model.parameters(), shared_average_model.parameters()):
    shared_average_param = args.trust_region_decay * shared_average_param + (1 - args.trust_region_decay) * shared_param


# Computes an "efficient trust region" loss (policy head only) based on an existing loss and two distributions
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold, g, k):
  kl = - (ref_distribution * (distribution.log() - ref_distribution.log())).sum(1).mean(0)

  # Compute dot products of gradients
  k_dot_g = (k * g).sum(1).mean(0)
  k_dot_k = (k ** 2).sum(1).mean(0)
  # Compute trust region update
  if k_dot_k.item() > 0:
    trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0).detach()
  else:
    trust_factor = torch.zeros(1)
  # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
  trust_loss = loss + trust_factor * kl

  return trust_loss


# Trains model
def _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret,
           average_policies, old_policies=None):
  off_policy = old_policies is not None
  action_size = policies[0].size(1)
  policy_loss, value_loss = 0, 0

  # Calculate n-step returns in forward view, stepping backwards from the last state
  t = len(rewards)
  for i in reversed(range(t)):
    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
    if off_policy:
      rho = policies[i].detach() / old_policies[i]
    else:
      rho = torch.ones(1, action_size)

    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]

    # Log policy log(π(a_i|s_i; θ))
    log_prob = policies[i].gather(1, actions[i]).log()
    # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
    single_step_policy_loss = -(rho.gather(1, actions[i]).clamp(max=args.trace_max) * log_prob * A.detach()).mean(
      0)  # Average over batch
    # Off-policy bias correction
    if off_policy:
      # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
      bias_weight = (1 - args.trace_max / rho).clamp(min=0) * policies[i]
      single_step_policy_loss -= (
                bias_weight * policies[i].log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(1).mean(0)
    if args.trust_region:
      # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
      k = -average_policies[i].gather(1, actions[i]) / (policies[i].gather(1, actions[i]) + 1e-10)
      if off_policy:
        g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-10).gather(1, actions[i]) \
             + (bias_weight * (Qs[i] - Vs[i].expand_as(Qs[i])) / (policies[i] + 1e-10)).sum(1)).detach()
      else:
        g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-10).gather(1, actions[
          i])).detach()
      # Policy update dθ ← dθ + ∂θ/∂θ∙z*
      policy_loss += _trust_region_loss(model, policies[i].gather(1, actions[i]) + 1e-10,
                                        average_policies[i].gather(1, actions[i]) + 1e-10, single_step_policy_loss,
                                        args.trust_region_threshold, g, k)
    else:
      # Policy update dθ ← dθ + ∂θ/∂θ∙g
      policy_loss += single_step_policy_loss

    # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
    policy_loss -= args.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(
      0)  # Sum over probabilities, average over batch

    # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
    Q = Qs[i].gather(1, actions[i])
    value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

    # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
    truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

  # Update networks
  _update_networks(args, T, model, shared_model, shared_average_model, policy_loss + value_loss, optimiser)


# Acts and trains model
def train1(rank, args, T, shared_model, shared_average_model, optimiser):
  # torch.manual_seed(args.seed + rank)
  # env = gym.make(args.env)
  # env.seed(args.seed + rank)

  print("training starts")

  env = TwoDGridWorld(4)

  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  if not args.on_policy:
    # Normalise memory capacity by number of training processes
    memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)


  t = 1  # Thread step counter
  done = True  # Start new episode
  epoch_length = 0

  while T.value() <= args.T_max:
    # print("inside while")
    # On-policy episode loop
    while True:
      # Sync with shared model at least every t_max steps
      model.load_state_dict(shared_model.state_dict())
      # Get starting timestep
      t_start = t

      # Reset or pass on hidden state
      if done:
        hx, avg_hx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        cx, avg_cx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        # print("env_reset", state)
        done, episode_length= False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

      while not done and t - t_start < args.t_max:
        # print("inside while not")
        # Calculate policy and values
        policy, Q, V, (hx, cx) = model(state, (hx, cx))
        average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))
        # Sample action
        action = torch.multinomial(policy, 1)[0, 0]
        # Step
        next_state, reward, done, _ = env.step(action.item()) # action = tensor(1); action.item() = 1
        # print("next_state", next_state)
        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        print("{}, Eps {},Epc{}, Step {}:".format(rank, T.value(),epoch_length,episode_length))
        episode_length += 1  # Increase episode counter
        if not args.on_policy:
          # Save (beginning part of) transition for offline training
          memory.append(state, action, reward, policy.detach())  # Save just tensors
        # Save outputs for online training
        [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                           (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]),
                                            average_policy))]
        # Increment counters
        t += 1
        T.increment()
        #Render the environment
        # env.render()
        # Update state
        state = next_state
      # print("outside while not")
      epoch_length += 1
      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        Qret = torch.zeros(1, 1)

        if not args.on_policy:
          # Save terminal state for offline training
          memory.append(state, None, None, None)
      else:
        # Qret = V(s_i; θ) for non-terminal s
        _, _, Qret, _ = model(state, (hx, cx))
        Qret = Qret.detach()

      # Train the network on-policy
      _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret,
             average_policies)

      # Finish on-policy episode
      if done:
        break


    # Train the network off-policy when enough experience has been collected
    if not args.on_policy and len(memory) >= args.replay_start:
      # print("training of off -policy started")
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        hx, avg_hx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)
        cx, avg_cx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
          action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
          reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
          old_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(state, (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                                             (policy, Q, V, action, reward, average_policy, old_policy))]

          # Unpack second half of transition
          next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1)

        # Do forward pass for all transitions
        _, _, Qret, _ = model(next_state, (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
               actions, rewards, Qret, average_policies, old_policies=old_policies)
    done = True


  env.close()
def train2(rank, args, T, shared_model, shared_average_model, optimiser):
  # torch.manual_seed(args.seed + rank)
  # time.sleep(80)
  # print("training starts")

  # device = iscuda()
  #
  # if device != "cpu":
  # 	cuda = True
  #
  # cuda = False

  env = TreatmentEnv()
  # env.seed(args.seed + rank)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  # print("model", model)
  model.train()

  if not args.on_policy:
    # Normalise memory capacity by number of training processes
    memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)
    # print("memory", memory)

  # # Where both the patients dose Matrix and Contours are loaded
  # # pid=('007','008','009','010','011','012','013','014','015','016')
  # pid = ['001']
  # # data_path=os.path.dirname(os.path.abspath('/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/007.hdf5'))
  # # data_path='/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/'
  # data_path = './lib_dvh/f_dijs/'
  # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/'
  # # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/'
  # for i in range(len(pid)):
  #   print("len(pid)", len(pid))
  #   globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')
  #   globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()['rectumLabel' + str(i)], \
  #   globals()['PTVLabel' + str(i)] = loadMask(data_path2 + str(pid[i]) + '.h5')
  #   print(globals()['doseMatrix_' + str(i)].shape)
  # # reward_check = zeros((MAX_EPISODES))
  # # q_check = zeros((MAX_EPISODES))
  # # vali_num = 0
  #
  # # Comment this when you have more than one test cases
  # testcase = 0
  # doseMatrix = globals()['doseMatrix_' + str(testcase)]
  # targetLabels = globals()['targetLabels_' + str(testcase)]
  # bladderLabel = globals()['bladderLabel' + str(testcase)]
  # rectumLabel = globals()['rectumLabel' + str(testcase)]
  # PTVLabel = globals()['PTVLabel' + str(testcase)]
  # MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel)

    # test_set=['12','17']#['01','07','08','09','10','11','12','13','14','15','16','17']
  test_set = ['01']
  # logging.info(
  #   '------------------------------------------ validation ----------------------------------------------------')
  # for sampleid in range(test_num):
  # config=get_config()
  # pid=('12','17')
  pid = ['01']

  # data_path = './lib_dvh/f_dijs/0'
  data_path = '/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/0'
  data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/0'
  # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/'

  for i in range(len(pid)):
    print("len(pid)", len(pid))
    globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')
    print("doseMatrix loaded")
    globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
      'rectumLabel' + str(i)], \
    globals()['PTVLabel' + str(i)] = loadMask(data_path2 + str(pid[i]) + '.h5')
    print("PTVLabel loaded")
    print(globals()['doseMatrix_' + str(i)].shape)

  # time.sleep(5)

  # reward_check = zeros((MAX_EPISODES))
  # q_check = zeros((MAX_EPISODES))
  # vali_num = 0

  # Comment this when you have more than one test cases
  testcase = 0
  doseMatrix = globals()['doseMatrix_' + str(testcase)]
  targetLabels = globals()['targetLabels_' + str(testcase)]
  bladderLabel = globals()['bladderLabel' + str(testcase)]
  rectumLabel = globals()['rectumLabel' + str(testcase)]
  PTVLabel = globals()['PTVLabel' + str(testcase)]
  MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel)


  t = 1  # Thread step counter
  done = True  # Start new episode


  while T.value() <= args.T_max:
    # On-policy episode loop
    # print("inside T.value() <= args.T_max:")

    while True:
      # Sync with shared model at least every t_max steps
      # print("loading model started")
      model.load_state_dict(shared_model.state_dict())
      # print("model loaded")
      # Get starting timestep
      t_start = t

      # Reset or pass on hidden state
      if done:
        # print("inside if done:")
        hx, avg_hx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        # print("hx created")
        cx, avg_cx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        # print("cx created")

        # Reset environment and done flag
        # state = state_to_tensor(env.reset())

        reward_sum_total = 0
        qvalue_sum = 0
        num_q = 0
        loss_perepoch = []
        doseMatrix = []
        targetLabels = []
        bladderLabel = []
        rectumLabel = []
        PTVLabel = []
        # env = TreatmentEnv()

        # Uncomment this when you have more than one test cases and then hit tab
        # for testcase in range (TRAIN_NUM):
        # 	logging.info('---------Training: Episode {}, Patient {}'.format(episode,testcase)+'-------------')
        # 	doseMatrix=globals()['doseMatrix_'+str(testcase)]
        # 	targetLabels=globals()['targetLabels_'+str(testcase)]
        # 	bladderLabel = globals()['bladderLabel'+str(testcase)]
        # 	rectumLabel	= globals()['rectumLabel'+str(testcase)]
        # 	PTVLabel = globals()['PTVLabel'+str(testcase)]
        # ------------------------ initial paramaters & input --------------------
        # tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC = 1, 1, 1, 1, 1, 1, 0.1, 1, 1
        epsilon = 1e-10
        tPTV = random.uniform(1+epsilon,1.2-epsilon)
        tBLA = random.uniform(0+epsilon,1-epsilon)
        tREC = random.uniform(0+epsilon,1-epsilon)
        lambdaPTV = random.uniform(0+epsilon,30-epsilon)
        lambdaBLA = random.uniform(0+epsilon,30-epsilon)
        lambdaREC = random.uniform(0+epsilon,30-epsilon)
        VPTV = random.uniform(0+epsilon,0.3-epsilon)
        VBLA = random.uniform(0+epsilon,1-epsilon)
        VREC = random.uniform(0+epsilon,1-epsilon)
        step_count = 0
        # --------------------- solve treatment planning optmization -----------------------------
        # MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels,bladderLabel, rectumLabel)
        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        # print("runOpt_dvh started")
        state, _, xVec = \
          runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,
                     VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)
        # print("runOpt_dvh ended")
        # state = np.array(state) #Please consider converting the list to a single numpy.ndarray with numpy.array()
        # state = torch.tensor([state], dtype=torch.float32)

        ################ Uncomment: For traning results #####################
        # tPTV_all = np.zeros((MAX_STEP + 1))
        # tBLA_all = np.zeros((MAX_STEP + 1))
        # tREC_all = np.zeros((MAX_STEP + 1))
        # lambdaPTV_all = np.zeros((MAX_STEP + 1))
        # lambdaBLA_all = np.zeros((MAX_STEP + 1))
        # lambdaREC_all = np.zeros((MAX_STEP + 1))
        # VPTV_all = np.zeros((MAX_STEP + 1))
        # VBLA_all = np.zeros((MAX_STEP + 1))
        # VREC_all = np.zeros((MAX_STEP + 1))
        #
        # tPTV_all[0] = tPTV
        # tBLA_all[0] = tBLA
        # tREC_all[0] = tREC
        # lambdaPTV_all[0] = lambdaPTV
        # lambdaBLA_all[0] = lambdaBLA
        # lambdaREC_all[0] = lambdaREC
        # VPTV_all[0] = VPTV
        # VBLA_all[0] = VBLA
        # VREC_all[0] = VREC
        #
        # array_list = []

        ################ Uncomment end: For training results #####################

        # if cuda:
        #   state = torch.from_numpy(state).float()
        #   state = state.to(device)

        state = state_to_tensor(state)
        # print("Training score")
        Score_fine, Score, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
        # print("Initial score:",Score)

        done, episode_length = False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

      while not done and t - t_start < args.t_max:
        # print("val:",tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)
        # print("inside while not done and t - t_start < args.t_max:")
        # Calculate policy and values
        policy, Q, V, (hx, cx) = model(state, (hx, cx))
        average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))

        # Sample action
        action = torch.multinomial(policy, 1)[0, 0]
        # print("action",action)
        j = 1 #some random number as it is not used

        # Step
        # next_state, reward, done, _ = env.step(action.item())
        next_state, reward, Score_fine, Score, done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec = env.step(
          action.item(), j, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV,
          lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter)
        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        # print("reward",reward)
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        print("Train:{}, {},{}\n".format(rank, T.value(), episode_length))
        episode_length += 1  # Increase episode counter

        if not args.on_policy:
          # Save (beginning part of) transition for offline training
          memory.append(state, action, reward, policy.detach())  # Save just tensors
        # Save outputs for online training
        [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                           (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]), average_policy))]

        # Increment counters
        t += 1

        T.increment()

        # Update state
        state = next_state
        # print("t", t)
        # print("diff.", t - t_start)
        # print("tmax", args.t_max)
        # print("done", done)

      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        Qret = torch.zeros(1, 1)
        # print("Qret",Qret)

        if not args.on_policy:
          # Save terminal state for offline training
          memory.append(state, None, None, None)
          # print("memory.appended")
      else:
        # Qret = V(s_i; θ) for non-terminal s
        _, _, Qret, _ = model(state, (hx, cx))
        Qret = Qret.detach()
        # print("Qret detached",Qret)

      # Train the network on-policy
      # print("on policy training started")
      _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies)
      # print("on policy training ended")

      # Finish on-policy episode
      if done:
        break

    # Train the network off-policy when enough experience has been collected
    if not args.on_policy and len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        hx, avg_hx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)
        cx, avg_cx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
          action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
          reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
          old_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(state, (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                                             (policy, Q, V, action, reward, average_policy, old_policy))]

          # Unpack second half of transition
          next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1)

        # Do forward pass for all transitions
        _, _, Qret, _ = model(next_state, (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
               actions, rewards, Qret, average_policies, old_policies=old_policies)

      print("replay ratio run")

    done = True

  env.close()

def gaussian(x, sigma, peak, mu):
    y = np.zeros(x.size)
    for i in range(x.size):
        gauss = peak*np.exp((-0.5 * ((x[i]-mu) / sigma) ** 2))
        y[i] = gauss

    return y

def perturbDVH(state_test, index):
    sigma = 1.0
    kernel_size = int(6 * sigma + 1)  # 6*sigma gives a wide enough kernel for most uses
    x = np.linspace(-3*sigma, 3*sigma, kernel_size)

    # 1D Gaussian kernel
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel

    Y = state_test.detach().numpy()
     
    Y = np.reshape(Y, (100, 3), order='F')


    y_ptv = Y[:,0] 
    y_bladder = Y[:,1] 
    y_rectum = Y[:,2] 

    if (index>=0 and index<100):
        y_ptvBlurr = convolve1d(y_ptv, gaussian_kernel)
        y_ptvIndBlurr = np.arange(0,100)
        y_ptvIndBlurr = gaussian(y_ptvIndBlurr, 0.7 , y_ptv[index], y_ptvIndBlurr[index])
        y_ptvFinal = y_ptv*(1- y_ptvIndBlurr) + y_ptvBlurr*y_ptvIndBlurr
        y_bladderFinal = y_bladder
        y_rectumFinal = y_rectum

    if (index>=100 and index<200):
        y_bladderBlurr = convolve1d(y_bladder, gaussian_kernel)
        y_bladderIndBlurr = np.arange(0,100)
        y_bladderIndBlurr = gaussian(y_bladderIndBlurr, 0.7 , y_bladder[int(index-100)], y_bladderIndBlurr[int(index-100)])
        y_bladderFinal = y_bladder*(1- y_bladderIndBlurr) + y_bladderBlurr*y_bladderIndBlurr
        y_ptvFinal = y_ptv
        y_rectumFinal = y_rectum

    if (index>=199 and index<300):
        y_rectumBlurr = convolve1d(y_rectum, gaussian_kernel)
        y_rectumIndBlurr = np.arange(0,100)
        y_rectumIndBlurr = gaussian(y_rectumIndBlurr, 0.7 , y_rectum[int(index-200)], y_rectumIndBlurr[int(index-200)])
        y_rectumFinal = y_rectum*(1- y_rectumIndBlurr) + y_rectumBlurr*y_rectumIndBlurr
        y_bladderFinal = y_bladder
        y_ptvFinal = y_ptv

    Yp = np.zeros((100, 3))

    Yp[:, 0] = y_ptvFinal
    Yp[:, 1] = y_bladderFinal
    Yp[:, 2] = y_rectumFinal

    perturbedState = np.reshape(Yp, (100*3,), order = 'F')

    return perturbedState



def loss(MPTV, MBLA, MREC, MBLA1, MREC1, args, done, doseMatrix, patientid, epoch, PTVLabel, bladderLabel, rectumLabel, cuda, paraMax, paraMin, paraMax_tPTV, paraMin_tPTV, paraMax_tOAR, paraMax_VOAR, paraMax_VPTV, First_state, gradient, rep, actionArray):

    env = TreatmentEnv()
    model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)


    model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
    model.eval()
    Ret = 0
    Reward_array = []
    Trial = np.zeros((INPUT_SIZE, 3))
    Trial = np.reshape(Trial, (100 * 3,), order='F')
    prob_grad = state_to_tensor(Trial)
    Init_state = state_to_tensor(Trial)
    while True:
        # Reset or pass on hidden state
        if done:
            # Sync with shared model every episode

            # print(shared_model.state_dict())

            # check_model = model.state_dict()
            # print("inside 'if done'")
            # print(model.state_dict())



            hx = torch.zeros(1, args.hidden_size)
            cx = torch.zeros(1, args.hidden_size)
            # Reset environment and done flag
            # state = state_to_tensor(env.reset())
            # if T.value() == 0:
            #     epoch = 0
            # else:
            #     epoch = T.value() - 1

            done, episode_length = False, 0
            reward_sum = 0

            # for sampleid in range(1):
            # sampleid = 0
            # id = test_set[sampleid]
            # doseMatrix = globals()['doseMatrix_' + str(sampleid)]
            env = TreatmentEnv()
            # initializing planScore with something greater that 6
            planScore = 8.5
            # ------------------------ initial paramaters & input --------------------
            # Lower limit of the following block(except of tPTV) is being updated =[0, 0.1, [0.5, 0.3, 0.1, 0.3, 0.3, 0.1, 0.2, 0.4], [0.7, 0.3, 0.1, 0.3, 0.3, 0.1, 0.2, 0.4], [0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.3, 0.3]]
            if patientid == 0:
                # planScore = 8.5
                # while (planScore >= 6):
                # this is the general block
                # epsilon = 1e-10
                TPPs = np.load("/data2/mainul1/results_CORS/scratch6_30StepsNewParamenters3indCriTime/dataWithPlanscoreRun/0tpptuning120499.npz")

                tPTV = TPPs['l1'][rep]
                tBLA = TPPs['l2'][rep]
                tREC = TPPs['l3'][rep]
                lambdaPTV = TPPs['l4'][rep]
                lambdaBLA = TPPs['l5'][rep]
                lambdaREC = TPPs['l6'][rep]
                VPTV = TPPs['l7'][rep]
                VBLA = TPPs['l8'][rep]
                VREC = TPPs['l9'][rep]
                xVec = np.ones((MPTV.shape[1],))
                gamma = np.zeros((MPTV.shape[0],))

                # # This is the very special box for testing TORTS dataset

                # tPTV = 1
                # tBLA = 1
                # tREC = 1
                # lambdaPTV = 1
                # lambdaBLA = 1
                # lambdaREC = 1
                # VPTV = 0.1
                # VBLA = 1
                # VREC = 1
                xVec = np.ones((MPTV.shape[1],))
                gamma = np.zeros((MPTV.shape[0],))
                print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
                    tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)
                # --------------------- solve treatment planning optmization -----------------------------
                state_test0, iter, xVec = \
                    runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC,
                               lambdaPTV, lambdaBLA, lambdaREC,
                               VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)
                # --------------------- generate input for NN -----------------------------
                Dose = doseMatrix.dot(xVec)
                DPTV = MPTV.dot(xVec)
                DBLA = MBLA.dot(xVec)
                DREC = MREC.dot(xVec)
                DPTV = np.sort(DPTV)
                DPTV = np.flipud(DPTV)
                DBLA = np.sort(DBLA)
                DBLA = np.flipud(DBLA)
                DREC = np.sort(DREC)
                DREC = np.flipud(DREC)
                # For plotting against Fixed edge_ptv
                # edge_ptv = np.zeros((100 + 1,))
                # edge_ptv[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                # x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                # (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                # y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

                # edge_bladder = np.zeros((100 + 1,))
                # edge_bladder[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                # x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                # (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                # y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

                # edge_rectum = np.zeros((100 + 1,))
                # edge_rectum[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                # x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                # (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                # y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

                # For plotting against changing and fixed both edge_ptv
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

                planscoresSavePath = '/data2/mainul/ExplainableAIResultsAllSteps/planscores/'
                os.makedirs(planscoresSavePath, exist_ok = True)
                data_result_path='/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'
                os.makedirs(data_result_path, exist_ok = True)
                np.save(data_result_path+str(patientid)+'xDVHYfull' + str(rep)  + 'step' + str(episode_length),
                    Y)
                np.save(data_result_path+str(patientid)+'xDVHY' + str(rep)  + 'step' + str(episode_length),
                    state_test0)
                # np.save(data_result_path+id+'xDVHYInitial',
                #         Y)
                # np.save(data_result_path+id+'xDVHXInitial',
                # np.save(data_result_path+id+'xDVHXInitial',
                #         X)
                # np.save(data_result_path+id+'xVecInitial', xVec)
                # np.save(data_result_path+id+'DoseInitial', Dose)
                np.save(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length), Dose)
                np.save(data_result_path+str(patientid)+'doseMatrix' + str(epoch)  + 'step' + str(episode_length), doseMatrix)
                np.save(data_result_path+str(patientid)+'bladderLabel' + str(epoch)  + 'step' + str(episode_length), bladderLabel)
                np.save(data_result_path+str(patientid)+'rectumLabel' + str(epoch)  + 'step' + str(episode_length), rectumLabel)
                np.save(data_result_path+str(patientid)+'PTVLabel' + str(epoch)  + 'step' + str(episode_length), PTVLabel)

                MAX_STEP = args.max_episode_length + 1
                tPTV_all = np.zeros((MAX_STEP + 1))
                tBLA_all = np.zeros((MAX_STEP + 1))
                tREC_all = np.zeros((MAX_STEP + 1))
                lambdaPTV_all = np.zeros((MAX_STEP + 1))
                lambdaBLA_all = np.zeros((MAX_STEP + 1))
                lambdaREC_all = np.zeros((MAX_STEP + 1))
                VPTV_all = np.zeros((MAX_STEP + 1))
                VBLA_all = np.zeros((MAX_STEP + 1))
                VREC_all = np.zeros((MAX_STEP + 1))
                planScore_all = np.zeros((MAX_STEP + 1))
                planScore_fine_all = np.zeros((MAX_STEP + 1))
                # print("Testing score")
                planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)
                # logging.info('---------------------- initialization ------------------------------')
                # logging.info("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(iter, planScore,planScore_fine))
                Score_fine, Score, _ = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)                                   

                tPTV_all[0] = tPTV
                tBLA_all[0] = tBLA
                tREC_all[0] = tREC
                lambdaPTV_all[0] = lambdaPTV
                lambdaBLA_all[0] = lambdaBLA
                lambdaREC_all[0] = lambdaREC
                VPTV_all[0] = VPTV
                VBLA_all[0] = VBLA
                VREC_all[0] = VREC
                planScore_all[0] = planScore
                planScore_fine_all[0] = planScore_fine
                print(planScore)
                np.save(planscoresSavePath+str(patientid)+ 'planscoreInsideIf',planScore)
                # ----------------two ways of NN schemes (with/without rules) --------------------------
                state_test = state_test0
                # print('state_test',state_test)

                if cuda:
                    state_test = torch.from_numpy(state_test).float()
                    state_test = state_test.to(device)
                 

                state_test = state_to_tensor(state_test)
                print('begining state', state_test)
                # stateI = torch.tensor(state_testI, requires_grad=True)
                # Ret = 0
                # Reward_array = []
                # prob_grad = 0
                # Init_state = state_test
                # ------------------------ initial paramaters & input --------------------
            else:
                state_test = perturbDVH(First_state, int(patientid-1))
                print('perturbed state', state_test)
                Y = state_test
                state_test = state_to_tensor(state_test)
                data_result_path='/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'
                planscoresSavePath = '/data2/mainul/ExplainableAIResultsAllSteps/planscores/'
                actionSavePath = '/data2/mainul/ExplainableAIResultsAllSteps/actions/'
                os.makedirs(actionSavePath, exist_ok = True)

                np.save(data_result_path+str(patientid)+'xDVHY' + str(rep)  + 'step' + str(episode_length),Y)

        # # Optionally render validation states
        # if args.render:
        #   env.render()
        # print('state_test_before_tensor',state_test)
        state_test = torch.tensor(state_test, requires_grad = True)
        # print('state_test_again',state_test)
        # Calculate policy
        # The next line is commented out to check if the gradient calculation becomes possible
        # with torch.no_grad():
        policy, _, _, (hx, cx) = model(state_test, (hx, cx))
        print('policy', policy)
        policySave = policy.detach().numpy()
        np.save(data_result_path+ str(patientid)+ 'policy'+ str(rep)+str(episode_length), policySave)

        # Choose action greedily
        # action = policy.max(1)[1][0]
        action = torch.multinomial(policy, 1)[0, 0]
        print('action', action)
        # a = torch.multinomial(policy,1)
        # print('action', a, 'patientid', patientid)
        # # action.backward()
        t = 1
        ###################################################################
        ####################################################################
        # _, reward, _ = env.step(action.item())
        if episode_length == 0:
            a = action
            print('action', a, 'patientid', patientid)
            np.save(data_result_path+str(patientid)+'actions' + str(rep)  + 'step' + str(episode_length),a.detach().numpy())
            actionArray[rep][patientid] = a.item()
            print('actionArray', actionArray)
            prob = policy[0,action]
            np.save(data_result_path+str(patientid)+'prob' + str(rep)  + 'step' + str(episode_length),prob.detach().numpy())
            print('prob',prob)
            torch.log(prob).backward()
            print('prob_grad', state_test.grad)
            prob_grad = state_test.grad
            np.save(data_result_path+str(patientid)+'prob_grad' + str(rep)  + 'step' + str(episode_length),prob_grad.detach().numpy())
            Init_state = state_test
            
        # prob_array += [prob]

        # The next lines are to checking if probability is right
        # print('prob', prob)
        # summ = 0
        # for i in range(18):
        #     prob = policy[0,i].item()
        #     summ += prob
        #     print('prob', prob)
        
        # print('sum', summ)
        # The try except clause is added in an attempt to store the DVHs and run 20 times in one attempt
        try:
            if action == 0:
                tPTV = min(tPTV * 1.01, paraMax_tPTV)
            elif action == 1:
                tPTV = max(tPTV * 0.91, paraMin_tPTV)
            elif action == 2:
                tBLA = min(tBLA * 1.25, paraMax_tOAR)
            elif action == 3:
                tBLA = tBLA * 0.6
            elif action == 4:
                tREC = min(tREC * 1.25, paraMax_tOAR)
            elif action == 5:
                tREC = tREC * 0.6
            elif action == 6:
                lambdaPTV = lambdaPTV * 1.65
            elif action == 7:
                lambdaPTV = lambdaPTV * 0.6
            elif action == 8:
                lambdaBLA = lambdaBLA * 1.65
            elif action == 9:
                lambdaBLA = lambdaBLA * 0.6
            elif action == 10:
                lambdaREC = lambdaREC * 1.65
            elif action == 11:
                lambdaREC = lambdaREC * 0.6
            elif action == 12:
                VPTV = min(VPTV * 1.25, paraMax_VPTV)
            elif action == 13:
                VPTV = VPTV * 0.8
            elif action == 14:
                VBLA = min(VBLA * 1.25, paraMax_VOAR)
            elif action == 15:
                VBLA = VBLA * 0.8
            elif action == 16:
                VREC = min(VREC * 1.25, paraMax_VOAR)
            elif action == 17:
                VREC = VREC * 0.8

        except Exception as e:
            print('error')
            break


        # print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
        #         tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)

        # --------------------- solve treatment planning optmization -----------------------------
        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        n_state_test, iter, xVec = \
            runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA,
                       VREC, xVec,
                       gamma, pdose, maxiter)
        # n_state_test.backward()
        # print('state_grad', state_test.grad)
        planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)

        Score_fine1, Score1, _ = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
        # Score_fine.backward()
        reward = (Score_fine1 - Score_fine) + (Score1 - Score) * 4
        # reward.backward()
        # reward_grad = state_test.grad
        # print('Reward Grad', reward_grad)
        print('reward',reward)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        print('clipped reward', reward)
        Ret += ((0.99)**episode_length)*reward
        Reward_array += [reward]
        print('return', Ret)
        Score_fine, Score, _ = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)


        print("Test:{},{} ,{}, Individual_scores[{}]".format(

                                                                                           epoch,
                                                                                           planScore,
                                                                                           planScore_fine,
                                                                                           scoreall))
        # state_test = tf.convert_to_tensor([n_state_test],dtype=tf.float32)
        state_test = n_state_test
        if cuda:
            state_test = torch.from_numpy(state_test).float()
            state_test = state_test.to(device)

        state_test = state_to_tensor(state_test)

        j = episode_length

        # collect the result in each iteration
        # print("tPTV",tPTV)
        # print("tPTV_all",tPTV_all)
        # print("episode_length",episode_length)
        tPTV_all[j + 1] = tPTV
        # print("tPTV_all", tPTV_all)
        tBLA_all[j + 1] = tBLA
        tREC_all[j + 1] = tREC
        lambdaPTV_all[j  + 1] = lambdaPTV
        lambdaBLA_all[j + 1] = lambdaBLA
        lambdaREC_all[j + 1] = lambdaREC
        VPTV_all[j + 1] = VPTV
        VBLA_all[j + 1] = VBLA
        VREC_all[j + 1] = VREC
        planScore_all[j + 1] = planScore
        planScore_fine_all[j + 1] = planScore_fine
        print(planScore)
        # print(tPTV_all,tBLA_all, tREC_all, lambdaPTV_all,lambdaBLA_all,lambdaREC_all,VPTV_all,VBLA_all, VREC_all,planScore_all,planScore_fine_all)

        # if paraidx == 0:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  tPTV: {} case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
        #                                                                                     round(tPTV,2), case, round(planScore_fine,3), round(planScore,3)))
        # if paraidx == 1:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  tBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
        #                                                                                    round(tBLA,2), case, round(planScore_fine,3), round(planScore,3)))
        # if paraidx == 2:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  tREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
        #                                                                                     round(tREC,2), case, round(planScore_fine,3), round(planScore,3)))
        # if paraidx == 3:
        #     logging.info(
        #         "Step: {}  Iteration: {}  Action: {}  lambdaPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter,
        #                  action, round(lambdaPTV,2),
        #                                                                                                                case,
        #                                                                                                                round(planScore_fine,3),
        #                                                                                                       round(planScore,3)))

        # if paraidx == 4:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  lambdaBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1,
        #                  iter, action,
        #                                                                                     round(lambdaBLA,2), case, round(planScore_fine,3),
        #                                                                                     round(planScore,3)))
        # if paraidx == 5:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  lambdaREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter,
        #                  action,
        #                                                                                     round(lambdaREC,2), case, round(planScore_fine,3),
        #                                                                                     round(planScore,3)))
        # if paraidx == 6:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  VPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
        #                                                                                     round(VPTV,4), case, round(planScore_fine,3), round(planScore,3)))
        # if paraidx == 7:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  VBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
        #                                                                                    round(VBLA,4), case, round(planScore_fine,3), round(planScore,3)))
        # if paraidx == 8:
        #     logging.info("Step: {}  Iteration: {}  Action: {}  VREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
        #                                                                                     round(VREC,4), case, round(planScore_fine,3), round(planScore,3)))

        Dose = doseMatrix.dot(xVec)
        # np.save(data_result_path+id+'xVec' + str(episode)  + 'step' + str(i + 1), xVec)
        # np.save(data_result_path+id+'xDose' + str(episode)  + 'step' + str(i + 1), Dose)
        np.save(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length + 1), Dose)

        DPTV = MPTV.dot(xVec)
        DBLA = MBLA.dot(xVec)
        DREC = MREC.dot(xVec)
        DPTV = np.sort(DPTV)
        DPTV = np.flipud(DPTV)
        DBLA = np.sort(DBLA)
        DBLA = np.flipud(DBLA)
        DREC = np.sort(DREC)
        DREC = np.flipud(DREC)
        # # For plotting against fixed
        # edge_ptv = np.zeros((100 + 1,))
        # edge_ptv[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
        # x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
        # (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
        # y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

        # edge_bladder = np.zeros((100 + 1,))
        # edge_bladder[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
        # x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
        # (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
        # y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

        # edge_rectum = np.zeros((100 + 1,))
        # edge_rectum[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
        # x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
        # (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
        # y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

        # Y = np.zeros((100, 6))
        # Y[:, 0] = y_ptv
        # Y[:, 1] = y_bladder
        # Y[:, 2] = y_rectum
        # # X = np.zeros((1000, 3))
        # Y[:, 3] = x_ptv
        # Y[:, 4] = x_bladder
        # Y[:, 5] = x_rectum

        # For plotting against changing and fixed both edge_ptv
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

        np.save(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length + 1),
                Y)
        # np.save(data_result_path+id+'xDVHX' + str(epoch) + 'step' + str(episode_length + 1),
        #         X)
        # np.save(data_result_path + id + 'xDVHY' + str(episode) + 'step' + str(i + 1),
        #         Y)
        # np.save(data_result_path + id + 'xDVHX' + str(episode) + 'step' + str(i + 1),
        #         X)


        # data_result_path2 = './data/data/Results/figuresPATp/'
        # plt.plot(x_ptv, y_ptv)
        # plt.plot(x_bladder, y_bladder)
        # plt.plot(x_rectum, y_rectum)
        # plt.legend(('ptv', 'bladder', 'rectum'))
        # plt.title('DVH' + str(epoch) + 'step' + str(episode_length))
        # plt.savefig(data_result_path2 + id + 'DVH' + str(epoch) + 'step' + str(episode_length) + '.png')
        # plt.show(block=False)
        # plt.close()

        check_model2 = model.state_dict()
        # print("outside 'if done'")
        # for m in range(model.state_dict()):
        #   if (check_model[m] == check_model2[m]).all():
        #     print("same file")

        # for x,y in zip(check_model.items(), check_model2.items()):
        #   if torch.eq(x,y):
        #     print("same file")

        # print("outside 'if done'")
        # print(model.state_dict())

        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        # print("Testing:  {}, Eps {}, Step {}:".format(rank, T.value(), episode_length))
        episode_length += 1  # Increase episode counter
        # print("Proceess {}, testing step {}:".format(rank, episode_length))
        np.save(planscoresSavePath+str(patientid)+'planscoreBeforeWhileBreaking'+str(episode_length),planScore)
        if planScore == 9:
            done = True
            break

        # Log and reset statistics at the end of every episode
        if done:
            # print("Tvalue below render", T.value())
            # print("testing:done")
            # Render the environment
            # args.evaluate = True
            # print("args.evaluate", args.evaluate)
            break
        # print("end of loop")
    # prob = policy[0,action].item()
    # Reward_array = torch.tensor(Reward_array, dtype=torch.float32)
    # The next five lines give return for all the states encountered but it is not necessary anymore
    # for index in range(len(Reward_array)):
    #     ret = 0
    #     for i in range(len(Reward_array)-index):
    #         ret+= Reward_array[i+index]*(0.99**i)
    #     Return+= [ret]

    # Convert lists to tensors
    # prob_tensor = torch.tensor(prob_array, dtype=torch.float32)
    # return_tensor = torch.tensor(Return, dtype=torch.float32)

    # loss = sum(a*b for a,b in zip(prob_tensor, return_tensor))

    # loss.backward()

    # gradient_of_loss = prob_tensor.grad
    # The next three steps are for calculating return again which is not necessary
    # Return = 0
    # for index, element in enumerate(Reward_array):
    #     Return += element*((0.99)**index)
    # The following two if else clause is added in an attempt to store the DVHs and run 20 times in one attempt.
    if patientid == 0:
        tpp_parameters = np.zeros((MAX_STEP + 1,11))
        tpp_parameters[:, 0] = np.array(tPTV_all)
        tpp_parameters[:, 1] = np.array(tBLA_all)
        tpp_parameters[:, 2] = np.array(tREC_all)
        tpp_parameters[:, 3] = np.array(lambdaPTV_all)
        tpp_parameters[:, 4] = np.array(lambdaBLA_all)
        tpp_parameters[:, 5] = np.array(lambdaREC_all)
        tpp_parameters[:, 6] = np.array(VPTV_all)
        tpp_parameters[:, 7] = np.array(VBLA_all)
        tpp_parameters[:, 8] = np.array(VREC_all)
        tpp_parameters[:, 9] = np.array(planScore_all)
        tpp_parameters[:, 10] = np.array(planScore_fine_all)
        # np.save(data_result_path + id + 'tpptuning' + str(epoch),
        #         tpp_parameters)
        name1 = data_result_path + str(patientid) + 'tpptuning' + str(epoch)
        np.savez(name1+'.npz',l1 = tPTV_all, l2 =tBLA_all,  l3 = tREC_all, l4 =lambdaPTV_all, l5 =lambdaBLA_all, l6 =lambdaREC_all, l7 = VPTV_all, l8= VBLA_all, l9 = VREC_all, l10 = planScore_all, l11 = planScore_fine_all)
        np.save(planscoresSavePath+str(patientid)+'planscoreBeforeForBreaking',planScore)

    if patientid == 0:
        print('Ret', Ret)
        return Init_state, -Ret*prob_grad



import matplotlib.pyplot as plt
def test(rank, args, T):
    print("testing starts")
    # torch.manual_seed(args.seed + rank)

    # device = iscuda()
    # cuda = False
    # if device != "cpu":
    #     cuda = True

    cuda = False

    save_dir = os.path.join('results_test', args.name)

    can_test = True  # Test flag
    t_start = 1  # Test step counter to check against global counter
    rewards, steps = [], []  # Rewards and steps for plotting
    l = str(len(str(args.T_max)))  # Max num. of digits for logging steps

    ###############################################################################################

    # patient_list=['01','07','08','09','10','11','12','13','14','15','16','17']
    test_set = patient_list
    # logging.info(
    #     '------------------------------------------ validation ----------------------------------------------------')
    # for sampleid in range(test_num):
    # config=get_config()
    # pid=('12','17')
    # pid = ['01']
    i = 0

    env = TreatmentEnv()
    model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)


    done = True  # Start new episode

    # stores step, reward, avg_steps and time
    results_dict = {'t': [], 'reward': [], 'avg_steps': [], 'time': []}

    # Loading of all MPTV and other files
    pid = patient_list
    # data_path = './lib_dvh/f_dijs/0'
    # The following two lines were the initial testing datapaths
    # data_path = '/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/0'
    # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/0'
    # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/'
    # This are the new UTSW testing dataset filepath
    # data_path = '/home/mainul1/Downloads/'
    # data_path2 = '/home/mainul1/Downloads/'
    # the following lines is for data_path for TORTS dataset
    # data_path = '/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/'
    # data_path2 = '/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/'
    # data_path_TORTS = '/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/Prostate_TORTS/'

    # This block is for testing with the CORS dataset
    for i in range(len(pid)):
        print("len(pid)", len(pid))
        print("loading patient:",pid[i])
        globals()['doseMatrix_' + str(i)] = loadDoseMatrix('test_onceagain.hdf5')
        # print("doseMatrix loaded")
        globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
            'rectumLabel' + str(i)], \
        globals()['PTVLabel' + str(i)] = loadMask("test_dose_mask_onceagain.h5","test_structure_mask_onceagain.h5",)
        print("PTVLabel loaded")

    # # This block is for the new utsw testing dataset
    # for i in range(len(pid)):
    #     print("len(pid)", len(pid))
    #     globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + '063' + '.hdf5')
    #     print("doseMatrix loaded")
    #     globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
    #       'rectumLabel' + str(i)], \
    #     globals()['PTVLabel' + str(i)] = loadMask(data_path2 + '063' + '.h5')
    #     print("PTVLabel loaded")
    #     print(globals()['doseMatrix_' + str(i)].shape)

    # This block is for TORTS dataset
    # pid = ['02', '05', '06', '09', '11']

    # for i in range(len(pid)):
    #     print("len(pid)", len(pid))
    #     print("loading patient:",pid[i])
    #     globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path_TORTS+ 'test_TORTS' + str(pid)+ '.hdf5')
    #     # print("doseMatrix loaded")
    #     globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
    #         'rectumLabel' + str(i)], \
    #     globals()['PTVLabel' + str(i)] = loadMask(data_path_TORTS + "test_dose_mask_TORTS" + str(pid) + ".h5", data_path_TORTS + "test_structure_mask_TORTS" + str(pid)".h5",)
    #     print("PTVLabel loaded")

    # # This block is for testing with UTSW dataset which is done to check if there has been any changes from the training to the testing
    # for i in range(len(pid)):
    #     print("len(pid)", len(pid))
    #     globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')
    #     print("doseMatrix loaded")
    #     globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
    #       'rectumLabel' + str(i)], \
    #     globals()['PTVLabel' + str(i)] = loadMask(data_path2 + str(pid[i]) + '.h5')
    #     print("PTVLabel loaded")
    #     print(globals()['doseMatrix_' + str(i)].shape)

        # print(globals()['doseMatrix_' + str(i)].shape)
        # print("testing T.value():", T.value())

    # reward_check = zeros((MAX_EPISODES))
    # q_check = zeros((MAX_EPISODES))
    # vali_num = 0

    # # Comment this when you have more than one test cases
    # testcase = 0
    # doseMatrix = globals()['doseMatrix_' + str(testcase)]
    # targetLabels = globals()['targetLabels_' + str(testcase)]
    # bladderLabel = globals()['bladderLabel' + str(testcase)]
    # rectumLabel = globals()['rectumLabel' + str(testcase)]
    # PTVLabel = globals()['PTVLabel' + str(testcase)]

    while T.value() <= args.T_max:
        if can_test:
            t_start = T.value()  # Reset counter

            # if T.value() == 0:
            #     epoch = 0
            # else:
            #     epoch = T.value() - 1
            epoch = 120499



            model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
            model.eval()
            # torch.save(model.state_dict(),
            #            os.path.join(save_dir,'episode' + str(epoch) + '.pth'))  # Save model params

            # ------------- range of parmaeter -----------------
            paraMax = 100000  # change in validation as well
            paraMin = 0.1
            paraMax_tPTV = 1.2
            paraMin_tPTV = 1
            paraMax_tOAR = 1
            paraMax_VOAR = 1
            paraMax_VPTV = 0.3
            # ---------------------------------------------------

            # for sampleid in range(2):


            # Evaluate over several episodes and average results
            avg_rewards, avg_episode_lengths = [], []
            # for patientid in range(args.evaluation_episodes):
            actionArray = np.zeros((1,301))
            data_result_path='/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'
            os.makedirs(data_result_path, exist_ok = True)

            SizeIndicator = np.load("/data2/mainul1/results_CORS/scratch6_30StepsNewParamenters3indCriTime/dataWithPlanscoreRun/0tpptuning120499.npz")

            repSize = SizeIndicator['l1'].nonzero()[0].size
            actionArray = np.zeros((int(repSize),301))

            for rep in range(int(repSize)):
                disRep = []
                discrepancy = 0
                print("========================================iteration", rep)
                for patientid in range(301):
                    id1 = patientid
                    print("========================================Patient",id1)
                    print("done", done)
                    # For CORS, put sampleid = 0 to get the same dataset.
                    sampleid = 0
                    doseMatrix = globals()['doseMatrix_' + str(sampleid)]
                    targetLabels = globals()['targetLabels_' + str(sampleid)]
                    bladderLabel = globals()['bladderLabel' + str(sampleid)]
                    rectumLabel = globals()['rectumLabel' + str(sampleid)]
                    PTVLabel = globals()['PTVLabel' + str(sampleid)]

                    MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel,
                                                                 rectumLabel)
                    # print("Epoch: {}, Evaluation number: {}",epoch, i)
                    # print("model.state_dict()",model.state_dict())
                    if patientid == 0:
                            Trial = np.zeros((INPUT_SIZE, 3))
                            Trial = np.reshape(Trial, (100 * 3,), order='F')
                            gradient = state_to_tensor(Trial)
                            First_state = state_to_tensor(Trial)

                    # The following if else clause is added in attempt of storing all the dvhs at once. initially it was only the if part
                    if patientid == 0:
                        First_state, gradient = loss(MPTV, MBLA, MREC, MBLA1, MREC1, args, done, doseMatrix, patientid, epoch, PTVLabel, bladderLabel, rectumLabel, cuda, paraMax, paraMin, paraMax_tPTV, paraMin_tPTV, paraMax_tOAR, paraMax_VOAR, paraMax_VPTV, First_state, gradient, rep, actionArray)
                        print('gradient', gradient, 'First_state', First_state)
                    else:
                        loss(MPTV, MBLA, MREC, MBLA1, MREC1, args, done, doseMatrix, patientid, epoch, PTVLabel, bladderLabel, rectumLabel, cuda, paraMax, paraMin, paraMax_tPTV, paraMin_tPTV, paraMax_tOAR, paraMax_VOAR, paraMax_VPTV, First_state, gradient, rep, actionArray)
                        continue

                        if patientid ==300:
                            break

                if actionArray[rep][0] != actionArray[rep][1]:
                    discrepancy += 1
                    disRep.append(rep)
                    print('Total discrepancy', discrepancy)
                    print('Iterations for discrepancy', disRep)


                if rep == 0:
                    np.save(data_result_path+ 'both'+'actions' + 'Total'  + 'step' + str(0), actionArray)
                    print('actionArray', actionArray)
                    np.save(data_result_path + 'TotalDiscrepancy', discrepancy)
                    np.save(data_result_path + 'IterationsForDiscrepancy', disRep)
# The line after this line is taken inside another method


                # while True:
                #     # Reset or pass on hidden state
                #     if done:
                #         # Sync with shared model every episode

                #         # print(shared_model.state_dict())

                #         # check_model = model.state_dict()
                #         # print("inside 'if done'")
                #         # print(model.state_dict())



                #         hx = torch.zeros(1, args.hidden_size)
                #         cx = torch.zeros(1, args.hidden_size)
                #         # Reset environment and done flag
                #         # state = state_to_tensor(env.reset())
                #         # if T.value() == 0:
                #         #     epoch = 0
                #         # else:
                #         #     epoch = T.value() - 1

                #         done, episode_length = False, 0
                #         reward_sum = 0

                #         # for sampleid in range(1):
                #         # sampleid = 0
                #         # id = test_set[sampleid]
                #         # doseMatrix = globals()['doseMatrix_' + str(sampleid)]
                #         env = TreatmentEnv()
                #         # initializing planScore with something greater that 6
                #         planScore = 8.5
                #         # ------------------------ initial paramaters & input --------------------
                #         # Lower limit of the following block(except of tPTV) is being updated =[0, 0.1, [0.5, 0.3, 0.1, 0.3, 0.3, 0.1, 0.2, 0.4], [0.7, 0.3, 0.1, 0.3, 0.3, 0.1, 0.2, 0.4], [0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.3, 0.3]]
                #         while (planScore >= 6):
                #             # this is the general block
                #             epsilon = 1e-10
                #             tPTV = random.uniform(1 + epsilon, 1.2 - epsilon)
                #             tBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             tREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             lambdaPTV = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             lambdaBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             lambdaREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             VPTV = random.uniform(0.1 + epsilon, 0.3 - epsilon)
                #             VBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             VREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                #             xVec = np.ones((MPTV.shape[1],))
                #             gamma = np.zeros((MPTV.shape[0],))

                #             # # This is the very special box for testing TORTS dataset

                #             # tPTV = 1
                #             # tBLA = 1
                #             # tREC = 1
                #             # lambdaPTV = 1
                #             # lambdaBLA = 1
                #             # lambdaREC = 1
                #             # VPTV = 0.1
                #             # VBLA = 1
                #             # VREC = 1
                #             xVec = np.ones((MPTV.shape[1],))
                #             gamma = np.zeros((MPTV.shape[0],))
                #             print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
                #                 tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)
                #             # --------------------- solve treatment planning optmization -----------------------------
                #             state_test0, iter, xVec = \
                #                 runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC,
                #                            lambdaPTV, lambdaBLA, lambdaREC,
                #                            VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)
                #             # --------------------- generate input for NN -----------------------------
                #             Dose = doseMatrix.dot(xVec)
                #             DPTV = MPTV.dot(xVec)
                #             DBLA = MBLA.dot(xVec)
                #             DREC = MREC.dot(xVec)
                #             DPTV = np.sort(DPTV)
                #             DPTV = np.flipud(DPTV)
                #             DBLA = np.sort(DBLA)
                #             DBLA = np.flipud(DBLA)
                #             DREC = np.sort(DREC)
                #             DREC = np.flipud(DREC)
                #             # For plotting against Fixed edge_ptv
                #             # edge_ptv = np.zeros((100 + 1,))
                #             # edge_ptv[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                #             # x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                #             # (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                #             # y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

                #             # edge_bladder = np.zeros((100 + 1,))
                #             # edge_bladder[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                #             # x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                #             # (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                #             # y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

                #             # edge_rectum = np.zeros((100 + 1,))
                #             # edge_rectum[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                #             # x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                #             # (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                #             # y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

                #             # For plotting against changing and fixed both edge_ptv
                #             edge_ptv = np.zeros((100 + 1,))
                #             edge_ptv_max = np.zeros((100 + 1,))                        
                #             edge_ptv[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                #             edge_ptv_max[1:100 + 1] = np.linspace(0, max(DPTV), 100)
                #             x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                #             (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                #             (n_ptv_max, b_max) = np.histogram(DPTV, bins=edge_ptv_max)
                #             y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
                #             y_ptv_max = 1 - np.cumsum(n_ptv_max / len(DPTV), axis=0)

                #             edge_bladder = np.zeros((100 + 1,))
                #             edge_bladder_max = np.zeros((100 + 1,))
                #             edge_bladder[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                #             edge_bladder_max[1:100 + 1] = np.linspace(0, max(DBLA), 100)                            
                #             x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                #             (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                #             (n_bladder_max, b_max) = np.histogram(DBLA, bins = edge_bladder_max)
                #             y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
                #             y_bladder_max = 1 - np.cumsum(n_bladder_max/len(DBLA), axis = 0)

                #             edge_rectum = np.zeros((100 + 1,))
                #             edge_rectum_max = np.zeros((100 + 1,))
                #             edge_rectum[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                #             edge_rectum_max[1:100 + 1] = np.linspace(0, max(DREC), 100)                            
                #             x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                #             (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                #             (n_rectum_max, b_max) = np.histogram(DREC , bins = edge_rectum_max)
                #             y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
                #             y_rectum_max = 1 - np.cumsum(n_rectum_max / len(DREC), axis = 0)

                #             Y = np.zeros((100, 12))
                #             Y[:, 0] = y_ptv
                #             Y[:, 1] = y_bladder
                #             Y[:, 2] = y_rectum

                #             # X = np.zeros((1000, 3))
                #             Y[:, 3] = x_ptv
                #             Y[:, 4] = x_bladder
                #             Y[:, 5] = x_rectum

                #             # storing max range histograms
                #             Y[:, 6] = y_ptv_max
                #             Y[:, 7] = y_bladder_max
                #             Y[:, 8] = y_rectum_max

                #             Y[:, 9] = edge_ptv_max[1:100+1]
                #             Y[:, 10] = edge_bladder_max[1:100+1]
                #             Y[:, 11] = edge_rectum_max[1:100+1]

                #             planscoresSavePath = '/data/mainul1/results_CORS/FGSM_Attack/scratch6_30StepsNewParamenters3/planscores/'
                #             data_result_path='/data/mainul1/results_CORS/FGSM_Attack/scratch6_30StepsNewParamenters3/dataWithPlanscoreRun/'
                #             np.save(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length),
                #                 Y)
                #             # np.save(data_result_path+id+'xDVHYInitial',
                #             #         Y)
                #             # np.save(data_result_path+id+'xDVHXInitial',
                #             # np.save(data_result_path+id+'xDVHXInitial',
                #             #         X)
                #             # np.save(data_result_path+id+'xVecInitial', xVec)
                #             # np.save(data_result_path+id+'DoseInitial', Dose)
                #             np.save(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length), Dose)
                #             np.save(data_result_path+str(patientid)+'doseMatrix' + str(epoch)  + 'step' + str(episode_length), doseMatrix)
                #             np.save(data_result_path+str(patientid)+'bladderLabel' + str(epoch)  + 'step' + str(episode_length), bladderLabel)
                #             np.save(data_result_path+str(patientid)+'rectumLabel' + str(epoch)  + 'step' + str(episode_length), rectumLabel)
                #             np.save(data_result_path+str(patientid)+'PTVLabel' + str(epoch)  + 'step' + str(episode_length), PTVLabel)

                #             MAX_STEP = args.max_episode_length + 1
                #             tPTV_all = np.zeros((MAX_STEP + 1))
                #             tBLA_all = np.zeros((MAX_STEP + 1))
                #             tREC_all = np.zeros((MAX_STEP + 1))
                #             lambdaPTV_all = np.zeros((MAX_STEP + 1))
                #             lambdaBLA_all = np.zeros((MAX_STEP + 1))
                #             lambdaREC_all = np.zeros((MAX_STEP + 1))
                #             VPTV_all = np.zeros((MAX_STEP + 1))
                #             VBLA_all = np.zeros((MAX_STEP + 1))
                #             VREC_all = np.zeros((MAX_STEP + 1))
                #             planScore_all = np.zeros((MAX_STEP + 1))
                #             planScore_fine_all = np.zeros((MAX_STEP + 1))
                #             # print("Testing score")
                #             planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)
                #             # logging.info('---------------------- initialization ------------------------------')
                #             # logging.info("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(iter, planScore,planScore_fine))
                #             Score_fine, Score, _ = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)                                   

                #             tPTV_all[0] = tPTV
                #             tBLA_all[0] = tBLA
                #             tREC_all[0] = tREC
                #             lambdaPTV_all[0] = lambdaPTV
                #             lambdaBLA_all[0] = lambdaBLA
                #             lambdaREC_all[0] = lambdaREC
                #             VPTV_all[0] = VPTV
                #             VBLA_all[0] = VBLA
                #             VREC_all[0] = VREC
                #             planScore_all[0] = planScore
                #             planScore_fine_all[0] = planScore_fine
                #             print(planScore)
                #             np.save(planscoresSavePath+str(patientid)+ 'planscoreInsideIf',planScore)
                #             # ----------------two ways of NN schemes (with/without rules) --------------------------
                #             state_test = state_test0
                #             if cuda:
                #                 state_test = torch.from_numpy(state_test).float()
                #                 state_test = state_test.to(device)
                             

                #             state_test = state_to_tensor(state_test, requires_grad= True)
                #             stateI = torch.tensor(state_testI, requires_grad=True)
                #             Ret = 0
                #             Reward_array = []
                #             prob_array = []
                #         # ------------------------ initial paramaters & input --------------------
                        

                #     # # Optionally render validation states
                #     # if args.render:
                #     #   env.render()

                #     # Calculate policy
                #     with torch.no_grad():
                #         policy, _, _, (hx, cx) = model(state_test, (hx, cx))

                #     # Choose action greedily
                #     # action = policy.max(1)[1][0]
                #     action = torch.multinomial(policy, 1)[0, 0]
                #     t = 1
                #     ###################################################################
                #     ####################################################################
                #     # _, reward, _ = env.step(action.item())
                #     prob = policy[0,action].item()
                #     prob_array += [prob]

                #     # The next lines are to checking if probability is right
                #     # print('prob', prob)
                #     # summ = 0
                #     # for i in range(18):
	            #     #     prob = policy[0,i].item()
	            #     #     summ += prob
	            #     #     print('prob', prob)
                    
                #     # print('sum', summ)

                #     if action == 0:
                #         tPTV = min(tPTV * 1.01, paraMax_tPTV)
                #     elif action == 1:
                #         tPTV = max(tPTV * 0.91, paraMin_tPTV)
                #     elif action == 2:
                #         tBLA = min(tBLA * 1.25, paraMax_tOAR)
                #     elif action == 3:
                #         tBLA = tBLA * 0.6
                #     elif action == 4:
                #         tREC = min(tREC * 1.25, paraMax_tOAR)
                #     elif action == 5:
                #         tREC = tREC * 0.6
                #     elif action == 6:
                #         lambdaPTV = lambdaPTV * 1.65
                #     elif action == 7:
                #         lambdaPTV = lambdaPTV * 0.6
                #     elif action == 8:
                #         lambdaBLA = lambdaBLA * 1.65
                #     elif action == 9:
                #         lambdaBLA = lambdaBLA * 0.6
                #     elif action == 10:
                #         lambdaREC = lambdaREC * 1.65
                #     elif action == 11:
                #         lambdaREC = lambdaREC * 0.6
                #     elif action == 12:
                #         VPTV = min(VPTV * 1.25, paraMax_VPTV)
                #     elif action == 13:
                #         VPTV = VPTV * 0.8
                #     elif action == 14:
                #         VBLA = min(VBLA * 1.25, paraMax_VOAR)
                #     elif action == 15:
                #         VBLA = VBLA * 0.8
                #     elif action == 16:
                #         VREC = min(VREC * 1.25, paraMax_VOAR)
                #     elif action == 17:
                #         VREC = VREC * 0.8

                #     # print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
                #     #         tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)

                #     # --------------------- solve treatment planning optmization -----------------------------
                #     xVec = np.ones((MPTV.shape[1],))
                #     gamma = np.zeros((MPTV.shape[0],))
                #     n_state_test, iter, xVec = \
                #         runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA,
                #                    VREC, xVec,
                #                    gamma, pdose, maxiter)
                #     planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)

                #     Score_fine1, Score1, _ = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
                #     reward = (Score_fine1 - Score_fine) + (Score1 - Score) * 4
                #     print('reward',reward)
                #     reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                #     print('clipped reward', reward)
                #     Ret += ((0.99)^episode_length)*reward
                #     Reward_array += [reward]
                #     print('return', Ret)
                #     Score_fine, Score, _ = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)


                #     print("Test:{},{} ,{}, Individual_scores[{}]".format(

                #                                                                                        epoch,
                #                                                                                        planScore,
                #                                                                                        planScore_fine,
                #                                                                                        scoreall))
                #     # state_test = tf.convert_to_tensor([n_state_test],dtype=tf.float32)
                #     state_test = n_state_test
                #     if cuda:
                #         state_test = torch.from_numpy(state_test).float()
                #         state_test = state_test.to(device)

                #     state_test = state_to_tensor(state_test)

                #     j = episode_length

                #     # collect the result in each iteration
                #     # print("tPTV",tPTV)
                #     # print("tPTV_all",tPTV_all)
                #     # print("episode_length",episode_length)
                #     tPTV_all[j + 1] = tPTV
                #     # print("tPTV_all", tPTV_all)
                #     tBLA_all[j + 1] = tBLA
                #     tREC_all[j + 1] = tREC
                #     lambdaPTV_all[j  + 1] = lambdaPTV
                #     lambdaBLA_all[j + 1] = lambdaBLA
                #     lambdaREC_all[j + 1] = lambdaREC
                #     VPTV_all[j + 1] = VPTV
                #     VBLA_all[j + 1] = VBLA
                #     VREC_all[j + 1] = VREC
                #     planScore_all[j + 1] = planScore
                #     planScore_fine_all[j + 1] = planScore_fine
                #     print(planScore)
                #     # print(tPTV_all,tBLA_all, tREC_all, lambdaPTV_all,lambdaBLA_all,lambdaREC_all,VPTV_all,VBLA_all, VREC_all,planScore_all,planScore_fine_all)

                #     # if paraidx == 0:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  tPTV: {} case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                #     #                                                                                     round(tPTV,2), case, round(planScore_fine,3), round(planScore,3)))
                #     # if paraidx == 1:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  tBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                #     #                                                                                    round(tBLA,2), case, round(planScore_fine,3), round(planScore,3)))
                #     # if paraidx == 2:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  tREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                #     #                                                                                     round(tREC,2), case, round(planScore_fine,3), round(planScore,3)))
                #     # if paraidx == 3:
                #     #     logging.info(
                #     #         "Step: {}  Iteration: {}  Action: {}  lambdaPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter,
                #     #                  action, round(lambdaPTV,2),
                #     #                                                                                                                case,
                #     #                                                                                                                round(planScore_fine,3),
                #     #                                                                                                       round(planScore,3)))

                #     # if paraidx == 4:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  lambdaBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1,
                #     #                  iter, action,
                #     #                                                                                     round(lambdaBLA,2), case, round(planScore_fine,3),
                #     #                                                                                     round(planScore,3)))
                #     # if paraidx == 5:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  lambdaREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter,
                #     #                  action,
                #     #                                                                                     round(lambdaREC,2), case, round(planScore_fine,3),
                #     #                                                                                     round(planScore,3)))
                #     # if paraidx == 6:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  VPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                #     #                                                                                     round(VPTV,4), case, round(planScore_fine,3), round(planScore,3)))
                #     # if paraidx == 7:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  VBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                #     #                                                                                    round(VBLA,4), case, round(planScore_fine,3), round(planScore,3)))
                #     # if paraidx == 8:
                #     #     logging.info("Step: {}  Iteration: {}  Action: {}  VREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                #     #                                                                                     round(VREC,4), case, round(planScore_fine,3), round(planScore,3)))

                #     Dose = doseMatrix.dot(xVec)
                #     # np.save(data_result_path+id+'xVec' + str(episode)  + 'step' + str(i + 1), xVec)
                #     # np.save(data_result_path+id+'xDose' + str(episode)  + 'step' + str(i + 1), Dose)
                #     np.save(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length + 1), Dose)

                #     DPTV = MPTV.dot(xVec)
                #     DBLA = MBLA.dot(xVec)
                #     DREC = MREC.dot(xVec)
                #     DPTV = np.sort(DPTV)
                #     DPTV = np.flipud(DPTV)
                #     DBLA = np.sort(DBLA)
                #     DBLA = np.flipud(DBLA)
                #     DREC = np.sort(DREC)
                #     DREC = np.flipud(DREC)
                #     # # For plotting against fixed
                #     # edge_ptv = np.zeros((100 + 1,))
                #     # edge_ptv[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                #     # x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                #     # (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                #     # y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

                #     # edge_bladder = np.zeros((100 + 1,))
                #     # edge_bladder[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                #     # x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                #     # (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                #     # y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

                #     # edge_rectum = np.zeros((100 + 1,))
                #     # edge_rectum[1:100 + 1] = np.linspace(0, pdose * 1.15, 100)
                #     # x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                #     # (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                #     # y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

                #     # Y = np.zeros((100, 6))
                #     # Y[:, 0] = y_ptv
                #     # Y[:, 1] = y_bladder
                #     # Y[:, 2] = y_rectum
                #     # # X = np.zeros((1000, 3))
                #     # Y[:, 3] = x_ptv
                #     # Y[:, 4] = x_bladder
                #     # Y[:, 5] = x_rectum

                #     # For plotting against changing and fixed both edge_ptv
                #     edge_ptv = np.zeros((100 + 1,))
                #     edge_ptv_max = np.zeros((100 + 1,))                        
                #     edge_ptv[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                #     edge_ptv_max[1:100 + 1] = np.linspace(0, max(DPTV), 100)
                #     x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                #     (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                #     (n_ptv_max, b_max) = np.histogram(DPTV, bins=edge_ptv_max)
                #     y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
                #     y_ptv_max = 1 - np.cumsum(n_ptv_max / len(DPTV), axis=0)

                #     edge_bladder = np.zeros((100 + 1,))
                #     edge_bladder_max = np.zeros((100 + 1,))
                #     edge_bladder[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                #     edge_bladder_max[1:100 + 1] = np.linspace(0, max(DBLA), 100)                            
                #     x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                #     (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                #     (n_bladder_max, b_max) = np.histogram(DBLA, bins = edge_bladder_max)
                #     y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
                #     y_bladder_max = 1 - np.cumsum(n_bladder_max/len(DBLA), axis = 0)

                #     edge_rectum = np.zeros((100 + 1,))
                #     edge_rectum_max = np.zeros((100 + 1,))
                #     edge_rectum[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                #     edge_rectum_max[1:100 + 1] = np.linspace(0, max(DREC), 100)                            
                #     x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                #     (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                #     (n_rectum_max, b_max) = np.histogram(DREC , bins = edge_rectum_max)
                #     y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
                #     y_rectum_max = 1 - np.cumsum(n_rectum_max / len(DREC), axis = 0)

                #     Y = np.zeros((100, 12))
                #     Y[:, 0] = y_ptv
                #     Y[:, 1] = y_bladder
                #     Y[:, 2] = y_rectum

                #     # X = np.zeros((1000, 3))
                #     Y[:, 3] = x_ptv
                #     Y[:, 4] = x_bladder
                #     Y[:, 5] = x_rectum

                #     # storing max range histograms                    
                #     Y[:, 6] = y_ptv_max
                #     Y[:, 7] = y_bladder_max
                #     Y[:, 8] = y_rectum_max

                #     Y[:, 9] = edge_ptv_max[1:100+1]
                #     Y[:, 10] = edge_bladder_max[1:100+1]
                #     Y[:, 11] = edge_rectum_max[1:100+1]

                #     np.save(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length + 1),
                #             Y)
                #     # np.save(data_result_path+id+'xDVHX' + str(epoch) + 'step' + str(episode_length + 1),
                #     #         X)
                #     # np.save(data_result_path + id + 'xDVHY' + str(episode) + 'step' + str(i + 1),
                #     #         Y)
                #     # np.save(data_result_path + id + 'xDVHX' + str(episode) + 'step' + str(i + 1),
                #     #         X)


                #     # data_result_path2 = './data/data/Results/figuresPATp/'
                #     # plt.plot(x_ptv, y_ptv)
                #     # plt.plot(x_bladder, y_bladder)
                #     # plt.plot(x_rectum, y_rectum)
                #     # plt.legend(('ptv', 'bladder', 'rectum'))
                #     # plt.title('DVH' + str(epoch) + 'step' + str(episode_length))
                #     # plt.savefig(data_result_path2 + id + 'DVH' + str(epoch) + 'step' + str(episode_length) + '.png')
                #     # plt.show(block=False)
                #     # plt.close()

                #     check_model2 = model.state_dict()
                #     # print("outside 'if done'")
                #     # for m in range(model.state_dict()):
                #     #   if (check_model[m] == check_model2[m]).all():
                #     #     print("same file")

                #     # for x,y in zip(check_model.items(), check_model2.items()):
                #     #   if torch.eq(x,y):
                #     #     print("same file")

                #     # print("outside 'if done'")
                #     # print(model.state_dict())

                #     done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
                #     # print("Testing:  {}, Eps {}, Step {}:".format(rank, T.value(), episode_length))
                #     episode_length += 1  # Increase episode counter
                #     # print("Proceess {}, testing step {}:".format(rank, episode_length))
                #     np.save(planscoresSavePath+str(patientid)+'planscoreBeforeWhileBreaking'+str(episode_length),planScore)
                #     if planScore == 9:
                #         done = True
                #         break

                #     # Log and reset statistics at the end of every episode
                #     if done:
                #         # print("Tvalue below render", T.value())
                #         # print("testing:done")
                #         # Render the environment
                #         # args.evaluate = True
                #         # print("args.evaluate", args.evaluate)
                #         break
                #     # print("end of loop")
                # # prob = policy[0,action].item()
                # Return = []
                # for index in range(len(reward)):
                #     ret = 0
                #     for i in range(len(reward)-index):
                #         ret+= reward[i+index]*(0.99^i)
                #     Return+= [ret]

                # loss = sum(a*b for a,b in zip(prob_array, Return))

                # gradient_of_loss = loss.backward()


                # tpp_parameters = np.zeros((MAX_STEP + 1,11))
                # tpp_parameters[:, 0] = np.array(tPTV_all)
                # tpp_parameters[:, 1] = np.array(tBLA_all)
                # tpp_parameters[:, 2] = np.array(tREC_all)
                # tpp_parameters[:, 3] = np.array(lambdaPTV_all)
                # tpp_parameters[:, 4] = np.array(lambdaBLA_all)
                # tpp_parameters[:, 5] = np.array(lambdaREC_all)
                # tpp_parameters[:, 6] = np.array(VPTV_all)
                # tpp_parameters[:, 7] = np.array(VBLA_all)
                # tpp_parameters[:, 8] = np.array(VREC_all)
                # tpp_parameters[:, 9] = np.array(planScore_all)
                # tpp_parameters[:, 10] = np.array(planScore_fine_all)
                # # np.save(data_result_path + id + 'tpptuning' + str(epoch),
                # #         tpp_parameters)
                # name1 = data_result_path + str(patientid) + 'tpptuning' + str(epoch)
                # np.savez(name1+'.npz',l1 = tPTV_all, l2 =tBLA_all,  l3 = tREC_all, l4 =lambdaPTV_all, l5 =lambdaBLA_all, l6 =lambdaREC_all, l7 = VPTV_all, l8= VBLA_all, l9 = VREC_all, l10 = planScore_all, l11 = planScore_fine_all)
                # np.save(planscoresSavePath+str(patientid)+'planscoreBeforeForBreaking',planScore)
                # # print(tPTV_all,tBLA_all, tREC_all, lambdaPTV_all,lambdaBLA_all,lambdaREC_all,VPTV_all,VBLA_all, VREC_all,planScore_all,planScore_fine_all)

# The line up until the previous line are taken inside another method

                # plt.legend(('tPTV', 'tBLA', 'tREC', 'lambdaPTV', 'lambdaBLA', 'lambdaREC', 'VPTV', 'VBLA', 'VREC'))
                # plt.title('TPP tuning steps')
                # plt.savefig('./data/data/Results/figuresPATp/tpptuning' + str(epoch) + '.png')
                # plt.show(block=False)
                # plt.close()
                # # Log and reset statistics at the end of every episode
                # if done:
                #   print("testing:done")
                #   avg_rewards.append(reward_sum)
                #   avg_episode_lengths.append(episode_length)
                #   break

            # print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
            #       datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            #       t_start,
            #       sum(avg_rewards) / args.evaluation_episodes,
            #       sum(avg_episode_lengths) / args.evaluation_episodes))
            # fields = [t_start, sum(avg_rewards) / args.evaluation_episodes, sum(avg_episode_lengths) / args.evaluation_episodes, str(datetime.now())]

            # Dumping the results in pickle format
            with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
                pickle.dump(results_dict, f)

            if args.evaluate:
                return

                # # Saving the data in csv format
                # with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
                #   writer = csv.writer(f)
                #   writer.writerow(fields)

            steps.append(t_start)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))  # Save model params

            can_test = False  # Finish testing

        else:
            # print("t_start", t_start)
            # print("diff:", T.value() - t_start)
            if T.value() - t_start >= args.evaluation_interval:
                can_test = True

        time.sleep(0.001)  # Check if available to test every millisecond

    # Dumping the results in pickle format

    with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
        pickle.dump(results_dict, f)
    env.close()

    ###############################################################################################33
    ####### Uncomment the code below which works perfectly
    ################################################################################
    #########################################################3
    # # -*- coding: utf-8 -*-
    # import os
    # import time
    # from datetime import datetime
    # import gym
    # import torch
    # import csv
    # import pickle
    #
    # from model import ActorCritic
    # from utils import state_to_tensor, plot_line
    #
    # import os
    # from pathlib import Path
    # import logging
    # import numpy as np
    # from numpy import zeros
    # import math as m
    # import json
    # import gym
    # from gym import Env
    # from gym.spaces import Discrete
    # from collections import deque
    # import matplotlib.pyplot as plt
    # from lib_dvh.data_prep import loadDoseMatrix, loadMask, ProcessDmat
    # from lib_dvh.score_calcu import planIQ_train
    # from lib_dvh.AC_VPTN import TreatmentEnv  # TreatmentEnv needs to be above the Agent
    # # from lib_dvh.AC_VPTN import Agent #Agent needs to be down from the TreatmentEnv
    # from lib_dvh.AC_networks import ActorCritic
    #
    # from lib_dvh.TP_DVH_algo import runOpt_dvh
    # # from lib_dvh.validation_AC import bot_play
    # # from lib_dvh.exalu_training_AC1 import exalu_training_AC1
    #
    # import torch
    # import torch.optim as optim
    # from lib_dvh.myconfig import *
    # from termcolor import colored
    # global episode
    #
    # import time
    #
    # pdose = 1  # target dose for PTV
    # maxiter = 40  # maximum iteration number for treatment planing optimization
    #
    # def test(rank, args, T, shared_model):
    #   # torch.manual_seed(args.seed + rank)
    #
    #   device = iscuda()
    #   cuda = False
    #   if device != "cpu":
    #     cuda = True
    #
    #   cuda = False
    #
    #   env = TreatmentEnv()
    #   # env.seed(args.seed + rank)
    #   model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
    #   model.eval()
    #
    #   save_dir = os.path.join('results', args.name)
    #
    #   can_test = True  # Test flag
    #   t_start = 1  # Test step counter to check against global counter
    #   rewards, steps = [], []  # Rewards and steps for plotting
    #   l = str(len(str(args.T_max)))  # Max num. of digits for logging steps

    # # Where both the patients dose Matrix and Contours are loaded
    # # pid=('007','008','009','010','011','012','013','014','015','016')
    # pid = ['001']
    # # data_path=os.path.dirname(os.path.abspath('/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/007.hdf5'))
    # # data_path='/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/'
    # data_path = './lib_dvh/f_dijs/'
    # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/'
    # # data_path2 = './data/data/dose_deposition3/plostate_dijs/f_masks/'
    # for i in range(len(pid)):
    #   print("len(pid)", len(pid))
    #   globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')
    #   globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()['rectumLabel' + str(i)], \
    #   globals()['PTVLabel' + str(i)] = loadMask(data_path2 + str(pid[i]) + '.h5')
    #   print(globals()['doseMatrix_' + str(i)].shape)
    # # reward_check = zeros((MAX_EPISODES))
    # # q_check = zeros((MAX_EPISODES))
    # # vali_num = 0
    #
    # # Comment this when you have more than one test cases
    # testcase = 0
    # doseMatrix = globals()['doseMatrix_' + str(testcase)]
    # targetLabels = globals()['targetLabels_' + str(testcase)]
    # bladderLabel = globals()['bladderLabel' + str(testcase)]
    # rectumLabel = globals()['rectumLabel' + str(testcase)]
    # PTVLabel = globals()['PTVLabel' + str(testcase)]
    # MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel)
    #
    # done = True  # Start new episode
    #
    # # stores step, reward, avg_steps and time
    # results_dict = {'t': [], 'reward': [], 'avg_steps': [], 'time': []}
    #
    # while T.value() <= args.T_max:
    #   if can_test:
    #     t_start = T.value()  # Reset counter
    #
    #     # Evaluate over several episodes and average results
    #     avg_rewards, avg_episode_lengths = [], []
    #     for _ in range(args.evaluation_episodes):
    #       while True:
    #         # Reset or pass on hidden state
    #         if done:
    #           # Sync with shared model every episode
    #           model.load_state_dict(shared_model.state_dict())
    #           hx = torch.zeros(1, args.hidden_size)
    #           cx = torch.zeros(1, args.hidden_size)
    #           # Reset environment and done flag
    #           # state = state_to_tensor(env.reset())
    #
    #           reward_sum_total = 0
    #           qvalue_sum = 0
    #           num_q = 0
    #           loss_perepoch = []
    #           doseMatrix = []
    #           targetLabels = []
    #           bladderLabel = []
    #           rectumLabel = []
    #           PTVLabel = []
    #           # env = TreatmentEnv()
    #
    #           # Uncomment this when you have more than one test cases and then hit tab
    #           # for testcase in range (TRAIN_NUM):
    #           # 	logging.info('---------Training: Episode {}, Patient {}'.format(episode,testcase)+'-------------')
    #           # 	doseMatrix=globals()['doseMatrix_'+str(testcase)]
    #           # 	targetLabels=globals()['targetLabels_'+str(testcase)]
    #           # 	bladderLabel = globals()['bladderLabel'+str(testcase)]
    #           # 	rectumLabel	= globals()['rectumLabel'+str(testcase)]
    #           # 	PTVLabel = globals()['PTVLabel'+str(testcase)]
    #           # ------------------------ initial paramaters & input --------------------
    #           tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC = 1, 1, 1, 1, 1, 1, 0.1, 1, 1
    #           step_count = 0
    #           # --------------------- solve treatment planning optmization -----------------------------
    #           # MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels,bladderLabel, rectumLabel)
    #           xVec = np.ones((MPTV.shape[1],))
    #           gamma = np.zeros((MPTV.shape[0],))
    #           state, _, xVec = \
    #             runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,
    #                        VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)
    #           # state = np.array(state) #Please consider converting the list to a single numpy.ndarray with numpy.array()
    #           # state = torch.tensor([state], dtype=torch.float32)
    #
    #           ################ Uncomment: For traning results #####################
    #           # tPTV_all = np.zeros((MAX_STEP + 1))
    #           # tBLA_all = np.zeros((MAX_STEP + 1))
    #           # tREC_all = np.zeros((MAX_STEP + 1))
    #           # lambdaPTV_all = np.zeros((MAX_STEP + 1))
    #           # lambdaBLA_all = np.zeros((MAX_STEP + 1))
    #           # lambdaREC_all = np.zeros((MAX_STEP + 1))
    #           # VPTV_all = np.zeros((MAX_STEP + 1))
    #           # VBLA_all = np.zeros((MAX_STEP + 1))
    #           # VREC_all = np.zeros((MAX_STEP + 1))
    #           #
    #           # tPTV_all[0] = tPTV
    #           # tBLA_all[0] = tBLA
    #           # tREC_all[0] = tREC
    #           # lambdaPTV_all[0] = lambdaPTV
    #           # lambdaBLA_all[0] = lambdaBLA
    #           # lambdaREC_all[0] = lambdaREC
    #           # VPTV_all[0] = VPTV
    #           # VBLA_all[0] = VBLA
    #           # VREC_all[0] = VREC
    #           #
    #           # array_list = []
    #
    #           ################ Uncomment end: For training results #####################
    #
    #           if cuda:
    #             state = torch.from_numpy(state).float()
    #             state = state.to(device)
    #
    #           state = state_to_tensor(state)
    #           print("device", device)
    #           Score_fine, Score, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
    #
    #           done, episode_length = False, 0
    #           reward_sum = 0
    #
    #         # Optionally render validation states
    #         if args.render:
    #           env.render()
    #
    #         # Calculate policy
    #         with torch.no_grad():
    #           policy, _, _, (hx, cx) = model(state, (hx, cx))
    #
    #         # Choose action greedily
    #         action = policy.max(1)[1][0]
    #         t = 1
    #
    #         # Step
    #         # state, reward, done, _ = env.step(action.item())
    #         state, reward, Score_fine, Score, done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec = env.step(
    #           action, t, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV,
    #           lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter)
    #         state = state_to_tensor(state)
    #         reward_sum += reward
    #         done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
    #         episode_length += 1  # Increase episode counter
    #
    #         # Log and reset statistics at the end of every episode
    #         if done:
    #           print("testing:done")
    #           avg_rewards.append(reward_sum)
    #           avg_episode_lengths.append(episode_length)
    #           break
    #
    #     print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
    #       datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
    #       t_start,
    #       sum(avg_rewards) / args.evaluation_episodes,
    #       sum(avg_episode_lengths) / args.evaluation_episodes))
    #     fields = [t_start, sum(avg_rewards) / args.evaluation_episodes,
    #               sum(avg_episode_lengths) / args.evaluation_episodes, str(datetime.now())]
    #
    #     # storing data in the dictionary.
    #     results_dict['t'].append(t_start)
    #     results_dict['reward'].append(sum(avg_rewards) / args.evaluation_episodes)
    #     results_dict['avg_steps'].append(sum(avg_episode_lengths) / args.evaluation_episodes)
    #     results_dict['time'].append(str(datetime.now()))
    #
    #     # Dumping the results in pickle format
    #     with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
    #       pickle.dump(results_dict, f)
    #
    #     # Saving the data in csv format
    #     with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
    #       writer = csv.writer(f)
    #       writer.writerow(fields)
    #
    #     if args.evaluate:
    #       return
    #
    #     rewards.append(avg_rewards)  # Keep all evaluations
    #     steps.append(t_start)
    #     plot_line(steps, rewards, save_dir)  # Plot rewards
    #     torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))  # Save model params
    #     can_test = False  # Finish testing
    #   else:
    #     if T.value() - t_start >= args.evaluation_interval:
    #       can_test = True
    #
    #   time.sleep(0.001)  # Check if available to test every millisecond
    #
    # # Dumping the results in pickle format
    # with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
    #   pickle.dump(results_dict, f)
    #
    # env.close()

    ############################################################




# Global counter
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value




#
# Plots min, max and mean + standard deviation bars of a population over time
def plot_line(xs, ys_population, save_dir):
  max_colour = 'rgb(0, 132, 180)'
  mean_colour = 'rgb(0, 172, 237)'
  std_colour = 'rgba(29, 202, 255, 0.2)'

  ys = torch.tensor(ys_population)
  ys_min = ys.min(1)[0].squeeze()
  ys_max = ys.max(1)[0].squeeze()
  ys_mean = ys.mean(1).squeeze()
  ys_std = ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = go.Scatter(x=xs, y=ys_max.numpy(), mode='lines', line=dict(color=max_colour, dash='dash'), name='Max')
  trace_upper = go.Scatter(x=xs, y=ys_upper.numpy(), mode='lines', marker=dict(color="#444"), line=dict(width=0), name='+1 Std. Dev.', showlegend=False)
  trace_mean = go.Scatter(x=xs, y=ys_mean.numpy(), mode='lines', line=dict(color=mean_colour), name='Mean')
  trace_lower = go.Scatter(x=xs, y=ys_lower.numpy(), mode='lines', marker=dict(color="#444"), line=dict(width=0), fill='tonexty', fillcolor=std_colour, name='-1 Std. Dev.', showlegend=False)
  trace_min = go.Scatter(x=xs, y=ys_min.numpy(), mode='lines', line=dict(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_mean, trace_upper, trace_lower, trace_min, trace_max],
    'layout': dict(title='Rewards',
                   xaxis={'title': 'Step'},
                   yaxis={'title': 'Average Reward'})
  }, filename=os.path.join(save_dir, 'rewards.html'), auto_open=False)


# # # Arguments to use for faster testing
# parser = argparse.ArgumentParser(description='ACER')
# parser.add_argument('--seed', type=int, default=123, help='Random seed')
# parser.add_argument('--num-processes', type=int, default=3, metavar='N', help='Number of training async agents (does not include single validation agent)')
# parser.add_argument('--T-max', type=int, default=40, metavar='STEPS', help='Number of training steps')
# # parser.add_argument('--T-max', type=int, default=100, metavar='STEPS', help='Number of training steps')
# parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
# # parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
# parser.add_argument('--max-episode-length', type=int, default=4, metavar='LENGTH', help='Maximum episode length')
# parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
# parser.add_argument('--model', type=str, default = './results/results/episode29.pth', metavar='PARAMS', help='Pretrained model (state dict)')
# parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
# parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
# parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
# parser.add_argument('--replay-start', type=int, default=20000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
# parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
# parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
# parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
# parser.add_argument('--trust-region', action='store_true', help='Use trust region')
# parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight decay rate')
# parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
# parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
# parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
# parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
# parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
# parser.add_argument('--batch-size', type=int, default=16, metavar='SIZE', help='Off-policy batch size')
# parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
# parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
# parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
# parser.add_argument('--evaluation-interval', type=int, default=2, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
# parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
# parser.add_argument('--render', action='store_true', help='Render evaluation agent')
# parser.add_argument('--name', type=str, default='results', help='Save folder')
# parser.add_argument('--env', type=str, default='CartPole-v1',help='environment name')

# Original arguments to use
parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-processes', type=int, default=3, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=250000, metavar='STEPS', help='Number of training steps')
# parser.add_argument('--T-max', type=int, default=100, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
# parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
# parser.add_argument('--max-episode-length', type=int, default=20, metavar='LENGTH', help='Maximum episode length')
# parser.add_argument('--max-episode-length', type=int, default=29, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--max-episode-length', type=int, default=1, metavar='LENGTH', help='Maximum episode length')

parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=20000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=16, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
# parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluate', action='store_false', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=500, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=30, metavar='N', help='Number of evaluation episodes to average over')
# parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='results', help='Save folder')
parser.add_argument('--env', type=str, default='CartPole-v1',help='environment name')





if __name__ == '__main__':
  # BLAS setup
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'

  # Setup
  args = parser.parse_args()
  # Creating directories.
  save_dir = os.path.join('results_test', args.name)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  print(' ' * 26 + 'Options')

  # Saving parameters
  with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
      f.write(k + ' : ' + str(v) + '\n')
  # args.env = 'CartPole-v1'  # TODO: Remove hardcoded environment when code is more adaptable
  # args.env = TwoDGridWorld(4)
  mp.set_start_method(platform.python_version()[0] == '3' and 'spawn' or 'fork')  # Force true spawning (not forking) if available
  torch.manual_seed(args.seed)
  T = Counter()  # Global shared counter
  gym.logger.set_level(gym.logger.ERROR)  # Disable Gym warnings

  # Create shared network
  # env = gym.make(args.env)
  # env = TwoDGridWorld(4)
  env = TreatmentEnv()
  # print("env.observation_space, env.action_space, args.hidden_size", env.observation_space, env.action_space,
  #       args.hidden_size)
  # print("observation_space.shape[0]", env.observation_space.shape[0])
  # print("action_space.n", env.action_space.n)
  # print("args.hidden_size", args.hidden_size)
  shared_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  # shared_model.share_memory()
  # if args.model and os.path.isfile(args.model):
  #   # Load pretrained weights
  #   shared_model.load_state_dict(torch.load(args.model))
  # # Create average network
  # shared_average_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  # shared_average_model.load_state_dict(shared_model.state_dict())
  # shared_average_model.share_memory()
  # for param in shared_average_model.parameters():
  #   param.requires_grad = False
  # # Create optimiser for shared network parameters with shared statistics
  # optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
  # optimiser.share_memory()
  env.close()

  fields = ['t', 'rewards', 'avg_steps', 'time']
  with open(os.path.join(save_dir, 'test_results.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(fields)


  # Start validation agent
  processes = []
  p = mp.Process(target=test, args=(0, args, T))
  p.start()
  p.join()
  print("args.evaluate",args.evaluate)