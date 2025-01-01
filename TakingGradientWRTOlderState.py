import numpy as np
import torch
import sys
import os
import re
import numpy as np
from torch import nn
from torch.nn import functional as F
from gym.spaces import Discrete

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
    # return policy

def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)

# test_set = ['014', '015', '023', '073', '098']
# test_set = ['015', '098']
test_set = ['098']

model = ActorCritic(np.zeros([300]), Discrete(18), 32)
model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
model.eval()

for patient in test_set:
    Y = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
    repSize = Y['l1'].nonzero()[0].size
    # FinalStateBaseline = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{repSize-1}.npy'))
    # output = model(FinalStateBaseline)
    print('patient', patient)
    hx0 = torch.zeros(1, 32)
    cx0 = torch.zeros(1, 32)
    hx = hx0.clone()
    cx = cx0.clone()
    h = (hx, cx)

    # for i in range(repSize-1):
    for i in range(3):
        print('i', i)
        if i ==2:
            state = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy'))
            
            state.requires_grad_(True)
            policy, _, _, h = model(state, h)
        else:
            stateNext = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy'))
            policy, _, _, h = model(stateNext, h)
        h = (hx0.clone(), cx0.clone())

    policy[0][0].backward()

    print(state.grad)