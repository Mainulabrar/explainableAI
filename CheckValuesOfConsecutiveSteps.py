import numpy as np
import torch
import sys
import os
import re
import numpy as np
from torch import nn
from torch.nn import functional as F
from gym.spaces import Discrete
import shap
from captum.attr import IntegratedGradients
from gym import Env
import argparse

# class ActorCritic(nn.Module):
#   def __init__(self, observation_space, action_space, hidden_size):
#     super(ActorCritic, self).__init__()
#     # self.state_size = observation_space.shape[0]
#     self.state_size = 300
#     self.action_size = action_space.n


#     self.fc1 = nn.Linear(self.state_size, hidden_size) # (2,32)
#     self.lstm = nn.LSTMCell(hidden_size, hidden_size) # (32,32)
#     # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True) # (32,32)
#     self.fc_actor = nn.Linear(hidden_size, self.action_size) # (32, 4)
#     self.fc_critic = nn.Linear(hidden_size, self.action_size) # (32,4)
#     # self.hx = torch.zeros(1, 32) 
#     # self.cx = torch.zeros(1, 32)
#     # h = (hx, cx)

#   def forward(self, x):
  
#     hx = torch.zeros(x.size(0), 32)
#     cx = torch.zeros(x.size(0), 32)
#     # hx = torch.zeros(1, 32)
#     # cx = torch.zeros(1, 32)
#     # h = (self.hx, self.cx)
#     h = (hx, cx)
#     # Here, x = state
#     x = F.relu(self.fc1(x))
#     h = self.lstm(x, h)  # h is (hidden state, cell state)
#     x = h[0]
#     policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
#     Q = self.fc_critic(x)
#     V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
#     # return policy, Q, V, h
#     return policy 
#     # return V

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


def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)

# test_set = [ '015', '098']
test_set = ['098']

parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')

if __name__ == '__main__':
  for patient in test_set:
      StepNumIndicator = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
      repSize = StepNumIndicator['l1'].nonzero()[0].size
      Value = []

      for i in range(repSize-1):
          # Y = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy")
          Y = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy")
          print(f'step{i}',Y)
          # model = ActorCritic(np.zeros([300]), Discrete(18), 32)
          env = TreatmentEnv()
          model = ActorCritic(env.observation_space, env.action_space, 32)

          # model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
          model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))

          model.eval()
          state = state_to_tensor(Y)
          # Value.append(model(state).item())
          hx = torch.zeros(1, 32)
          cx = torch.zeros(1, 32)
          with torch.no_grad():
            policy, _, _, (hx, cx) = model(state, (hx, cx))
          # policy = model(state)
          print('patient', patient)
          print('i', i)
          print('policyHere', policy)
          print('PolicySaved', np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy'))
          print('PolicySavedDebug', np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy'))

    # print(Value)