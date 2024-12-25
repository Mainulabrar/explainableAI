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

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    # self.state_size = observation_space.shape[0]
    self.state_size = 300
    self.action_size = action_space.n


    self.fc1 = nn.Linear(self.state_size, hidden_size) # (2,32)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size) # (32,32)
    # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True) # (32,32)
    self.fc_actor = nn.Linear(hidden_size, self.action_size) # (32, 4)
    self.fc_critic = nn.Linear(hidden_size, self.action_size) # (32,4)
    # self.hx = torch.zeros(1, 32) 
    # self.cx = torch.zeros(1, 32)
    # h = (hx, cx)

  def forward(self, x):
  
    hx = torch.zeros(x.size(0), 32)
    cx = torch.zeros(x.size(0), 32)
    # hx = torch.zeros(1, 32)
    # cx = torch.zeros(1, 32)
    # h = (self.hx, self.cx)
    h = (hx, cx)
    # Here, x = state
    x = F.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    # return policy, Q, V, h
    return policy 

def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)

def backgroundForShap():
    ShapData = []
    hx = torch.zeros(1, 32)
    cx = torch.zeros(1, 32)
    for i in range(18):
        pattern = rf'^0xDVHY({i})step0\.npy$'
        for filename in os.listdir('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'):
            matches = re.match(pattern, filename)
            if matches:
                # fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
                fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/',filename)
                # state = state_to_tensor(np.load(fullname))
                ShapData.append(np.load(fullname))
                # ShapData.append(state)


    ShapData = np.array(ShapData)
    # ShapData = tuple(ShapData)
    # print(ShapData)
    ShapData = torch.from_numpy(ShapData)
    # ShapData = torch.from_numpy(ShapData).float().unsqueeze(0)
    # ShapData = state_to_tensor(ShapData)
    # ShapData = torch.stack(ShapData)
    # print('ShapData', ShapData)

    # hxBackground = hx.repeat(ShapData.size(0),1)
    # cxBackground = cx.repeat(ShapData.size(0),1)
    # hBackground = (hxBackground.float(), cxBackground.float())

    # background = (ShapData.float(), hBackground)
    trivialBackground = np.zeros((18,300))
    trivialBackground = torch.from_numpy(trivialBackground)

    # return ShapData.float()
    return trivialBackground.float()
    # return ShapData.float()
    # return background

model = ActorCritic(np.zeros([300]), Discrete(18), 32)
model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
model.eval()
background = backgroundForShap()
# print(background)
print(background.shape)
# state = state_to_tensor(np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0xDVHY0step0.npy'))
# output = model(*background)
output = model(background)
explainer = shap.GradientExplainer(model, background)
# print(output)
print(np.mean(output.detach().numpy(), axis = 0))
# print('shap Values For state 0', explainer.shap_values(state).shape)
# print('shap Values For state 0', np.sum(explainer.shap_values(state), axis = 1))
# print('adding Up baseline and summed all the feature contributions of shap values', np.mean(output.detach().numpy(), axis = 0)+ np.sum(explainer.shap_values(state), axis = 1))
# print('policy For state 0', model(state))
baseline = np.zeros(300)
baselineTensor = state_to_tensor(baseline)
print('baselineTensor', baselineTensor)
ig = IntegratedGradients(model)

# test_set = ['014', '015', '023', '073', '098']
test_set = ['015', '098']

for patient in test_set:
    Y = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
    repSize = Y['l1'].nonzero()[0].size
    # FinalStateBaseline = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{repSize-1}.npy'))
    optimizedFromTrivial = state_to_tensor(np.loadtxt('/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/BackgroundOptimizedSimplyFromZero.txt'))
    output = model(optimizedFromTrivial)

    for i in range(repSize-1):
        state = state_to_tensor(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy'))
        print(state.shape)
        ig = IntegratedGradients(model)
        Policy = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy")
        PolicySort = np.sort(Policy, axis = 1)[:, ::-1]
        actionPolicy = np.argwhere(Policy == PolicySort[0,0])[0,1].item()
        actionPolicy2nd = np.argwhere(Policy == PolicySort[0,1])[0,1].item()
        # actionPolicy = np.argmax(np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy")).item()
        print('actionPolicy', actionPolicy)
        # print('actionPolicy', actionPolicy2nd)
        # attributions, delta = ig.attribute(state, baselines= baselineTensor, target = actionPolicy, return_convergence_delta=True)
        # attributions, delta = ig.attribute(state, baselines= baselineTensor, target = actionPolicy2nd, return_convergence_delta=True)
        AllAttributionTensor = torch.zeros(18, 300)
        for j in range(18):
            # attributions, delta = ig.attribute(state, baselines= FinalStateBaseline, target = i, return_convergence_delta=True)
            attributions, delta = ig.attribute(state, baselines= optimizedFromTrivial, target = i, return_convergence_delta=True)
            # attributions, delta = ig.attribute(state, baselines= baselineTensor, target = i, return_convergence_delta=True)
            # The next 3 line block is when I want to sum the effect of contributing as well as suppressing===============================
            if i != actionPolicy:
                # attributions = -1* attributions
                attributions = -(1/17)* attributions
            AllAttributionTensor[i,:] = attributions.detach()[0,:]
            #=====================================================================================
            # The next 3 line block is when I want to consider only suppressing=======================================

        SummedAttributions = torch.sum(AllAttributionTensor, dim = 0)
        print(f'IG Values For state 0: step{i}', SummedAttributions)
        print(f'Total  IG Values For state 0: step{i}', np.sum(SummedAttributions.numpy(), axis = 0))
        print('adding Up baseline and summed all the feature contributions of IG values', output.detach().numpy()[0, actionPolicy]+ np.sum(SummedAttributions.numpy(), axis = 0))
        prediction = output.detach().numpy()[0, actionPolicy]+ np.sum(SummedAttributions.numpy(), axis = 0)
        print('Mannually Calculated delta', -model(state).detach().numpy()[0, actionPolicy] + prediction)

        # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionRep{i}.txt', attributions)
        # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attribution2ndRep{i}.txt', attributions)
        # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyFinalStateBaselineRep{i}.txt', SummedAttributions)
        # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyTrivialBaselineRep{i}.txt', SummedAttributions)
        # np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyOptimizedFromTrivialBaselineRep{i}.txt', SummedAttributions)
        np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyAveOptimizedFromTrivialBaselineRep{i}.txt', SummedAttributions)
        
        print("Attributions:", SummedAttributions.shape)
        # print("Convergence Delta:", delta)



# for i in range(18):
#     state = state_to_tensor(np.load(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0xDVHY{i}step0.npy'))
#     shap_values = explainer.shap_values(state)
#     # print(output)
#     # print('shap_values' ,shap_values)
#     print('shap_values shape', shap_values.shape)
#     # print('expectedValue', explainer.expected_value)
#     np.save(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0ShapValuestep{i}', shap_values)

# # print(np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0xDVHY0step0.npy'))



