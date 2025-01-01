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


def allGradient(state, hx, cx, action):
    model = ActorCritic(np.zeros([300]), Discrete(18), 32)
    model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
    model.eval()

    state = state_to_tensor(state)
    state.requires_grad_(True)
    Policy, _, _,( _, _ ) = model(state, (hx, cx))
    # print('PolicyNow', Policy)

    Policy[0][action].backward()
    
    grad = state.grad.clone()
    state.grad.zero_()
    return grad


def calculatingIG(InitState, Finalstate, Interval, F, hx0, cx0, hxf, cxf, action):
    alpha = 1.00/Interval

    allGrad = np.zeros(300)
    for i in range(Interval):
        state = InitState + i*alpha*(Finalstate-InitState)
        # print('state', {i}, state)
        hx = hx0 + i*alpha*(hxf-hx0)
        cx = cx0 + i*alpha*(cxf-cx0)
        print( 'i', i)
        grad = F(state, hx, cx, action)[0].numpy()
        allGrad = allGrad+ grad

    allDimensionGrad = alpha*allGrad
            
    return (Finalstate- InitState)*allDimensionGrad

# test_set = ['014', '015', '023', '073', '098']
# test_set = ['015', '098']
test_set = ['098']
baseline = np.zeros(300)
# baselineTensor = state_to_tensor(baseline)


for patient in test_set:
    Y = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
    repSize = Y['l1'].nonzero()[0].size
    # FinalStateBaseline = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{repSize-1}.npy'))
    # output = model(FinalStateBaseline)
    # if i != 0:
    net = 120499
    hx = torch.tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}hx120499step{0}.npy'))
    cx = torch.tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}cx120499step{0}.npy'))
    state = np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}xDVHY120499step{1}.npy')

    Policy = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}Policy120499step{1}.npy")
    print('savedPolicy', Policy)
    PolicySort = np.sort(Policy, axis = 1)[:, ::-1]
    actionPolicy = np.argwhere(Policy == PolicySort[0,0])[0,1].item()

    print('patient', patient)
    hx0 = torch.zeros(1, 32)
    cx0 = torch.zeros(1, 32)
    # hx = hx0.clone()
    # cx = cx0.clone()
    h = (hx, cx)

    print('gradient', allGradient(state, hx, cx, actionPolicy))
    print('attributes', calculatingIG(baseline, state, 1000, allGradient, hx0, cx0, hx, cx, actionPolicy))


# The next part is for multiple steps =======================================================
    # # for i in range(repSize-1):
    # for i in range(3):
    #     print('i', i)
    #     if i ==2:
    #         state = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy'))
            
    #         state.requires_grad_(True)
    #         policy, _, _, h = model(state, h)
    #     else:
    #         stateNext = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy'))
    #         policy, _, _, h = model(stateNext, h)
    #     h = (hx0.clone(), cx0.clone())

    # policy[0][0].backward()

    # print(state.grad)
# ======================================================================================

