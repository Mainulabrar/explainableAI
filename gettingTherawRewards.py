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
from contextlib import redirect_stdout
import sys
sys.path.append('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/')


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


NpzFile = np.load("/data2/mainul1/results_CORS/scratch6_30StepsNewParamenters3indCriTime/dataWithPlanscoreRun/0tpptuning120499.npz")
scores = NpzFile['l10']
scorefines = NpzFile['l11']

allrewards = []
for i in range(17):
	reward = (scorefines[i+1] - scorefines[i]) + (scores[i+1] - scores[i]) * 4
	print(f'{i}:',reward)
	allrewards.append(reward)

# np.savetxt('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0Allrewards.txt',allrewards)

