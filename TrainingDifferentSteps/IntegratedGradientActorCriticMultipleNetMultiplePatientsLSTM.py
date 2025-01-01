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

  def forward(self, x, hx, cx):

    # Here, x = state
    x = F.relu(self.fc1(x))
    h = self.lstm(x, (hx,cx))  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    # return policy, Q, V, h
    return policy

def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)


model = ActorCritic(np.zeros([300]), Discrete(18), 32)
model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
model.eval()
# background = backgroundForShap()

baseline = np.zeros(300)
baselineSharp = baseline.copy()
baselineSharp[0] = 95.0
baselineSharp[1] = 0.5
baselineSharp[2] = 0.05


baselineTensor = state_to_tensor(baseline)
print('baselineTensor', baselineTensor)
ig = IntegratedGradients(model)

test_set = ['014', '015', '023', '073', '098']
# test_set = ['015', '098']
# test_set = ['098']
NetSet = ['0', '19999', '39999', '59999', '79999', '99999', '119999', '139999', '159999', '179999', '199999', '219999', '223999']

for net in NetSet:
    model = ActorCritic(np.zeros([300]), Discrete(18), 32)
    model.load_state_dict(torch.load(f'/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode{net}.pth'))
    model.eval()
    for patient in test_set:
        Y = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/dataWithPlanscoreRun/{patient}tpptuning120499.npz")
        repSize = Y['l1'].nonzero()[0].size
        # FinalStateBaseline = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{repSize-1}.npy'))
        # output = model(FinalStateBaseline)
        print('StepNumber', repSize)
        print('patient', patient)
        hx0 = torch.zeros(1, 32)
        cx0 = torch.zeros(1, 32)
        hx = hx0.clone()
        cx = cx0.clone()
        h = (hx, cx)
        output = model(baselineTensor, hx, cx)
        np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/backgroundForIGLSTM.txt', output.detach().numpy())

        # for i in range(repSize-1):
        for i in range(repSize-1):
            print('i', i)
            state = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy'))
            if i != 0:
               hx = torch.tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/dataWithPlanscoreRun/{patient}hx120499step{i-1}.npy'))
               cx = torch.tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/dataWithPlanscoreRun/{patient}cx120499step{i-1}.npy'))
            print(state.shape)
            ig = IntegratedGradients(model)
            Policy = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy")
            print('savedPolicy', Policy)
            PolicySort = np.sort(Policy, axis = 1)[:, ::-1]
            actionPolicy = np.argwhere(Policy == PolicySort[0,0])[0,1].item()
            actionPolicy2nd = np.argwhere(Policy == PolicySort[0,1])[0,1].item()
            # actionPolicy = np.argmax(np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy")).item()
            print('actionPolicy', actionPolicy)
            # print('actionPolicy', actionPolicy2nd)
            attributions, delta = ig.attribute(inputs = (state, hx, cx), baselines= (baselineTensor,hx0, cx0), target = actionPolicy, return_convergence_delta=True)
            # attributions, delta = ig.attribute(inputs = (state, hx, cx), baselines= (baselineTensor,hx0, cx0), target = actionPolicy2nd, return_convergence_delta=True)
            # attributions, delta = ig.attribute(state, baselines= baselineTensor, target = actionPolicy2nd, return_convergence_delta=True)
            # attributions, delta = ig.attribute(state, baselines= FinalStateBaseline, target = actionPolicy, return_convergence_delta=True)

            print(f'IG Values For state 0: step{i}', attributions)
            TotalAttributions = np.sum(attributions[0].numpy())+np.sum(attributions[1].numpy())+np.sum(attributions[2].numpy())
            print(f'Total  IG Values For state 0: step{i}', TotalAttributions)
            prediction = output.detach().numpy()[0, actionPolicy]+ TotalAttributions
            # prediction = output.detach().numpy()[0, actionPolicy2nd]+ TotalAttributions        
            print('adding Up baseline and summed all the feature contributions of IG values', prediction)
            
            prediction = output.detach().numpy()[0, actionPolicy]+ TotalAttributions
            # prediction = output.detach().numpy()[0, actionPolicy2nd]+ TotalAttributions  

            print('Mannually Calculated delta', -model(state, hx, cx).detach().numpy()[0, actionPolicy] + prediction)
            print('Mannually Calculated delta from saved Policy', -Policy[0, actionPolicy] + prediction)        
            # print('Mannually Calculated delta', -model(state, hx, cx).detach().numpy()[0, actionPolicy2nd] + prediction)
            # print('Mannually Calculated delta from saved Policy', -Policy[0, actionPolicy2nd] + prediction)  

            np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/{patient}attributionStateRep{i}.txt', attributions[0].numpy())
            np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/{patient}attributionhxRep{i}.txt', attributions[1].numpy())
            np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/{patient}attributioncxRep{i}.txt', attributions[2].numpy())

            # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/{patient}attributionStateRep2nd{i}.txt', attributions[0].numpy())
            # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/{patient}attributionhxRep2nd{i}.txt', attributions[1].numpy())
            # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/{patient}attributioncxRep2nd{i}.txt', attributions[2].numpy())

            # # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attribution2ndRep{i}.txt', attributions)
            # # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionFinalStateBaselineRep{i}.txt', attributions)
            # print("Attributions:", attributions.shape)
            print("Convergence Delta:", delta)



# for i in range(18):
#     state = state_to_tensor(np.load(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0xDVHY{i}step0.npy'))
#     shap_values = explainer.shap_values(state)
#     # print(output)
#     # print('shap_values' ,shap_values)
#     print('shap_values shape', shap_values.shape)
#     # print('expectedValue', explainer.expected_value)
#     np.save(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0ShapValuestep{i}', shap_values)

# # print(np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0xDVHY0step0.npy'))



