import torch
import torch
import sys
import os
import re
import numpy as np
from torch import nn
from torch.nn import functional as F
from gym.spaces import Discrete
import matplotlib.pyplot as plt

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

model = ActorCritic(np.zeros([300]), Discrete(18), 32)
model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
model.eval()

# Initialize 300 points and the 18-dimensional tensor
baseline = np.zeros(300)
# baseline = np.random.rand(300)
baselineTensor = state_to_tensor(baseline)
baselineTensor.requires_grad_()  # 300 points with gradient tracking
target_value = 1 / 18  # Target value for each element in the 18-dimensional tensor

# Mock function for generating the 18-dimensional tensor from points
# Replace this with your actual function
def compute_tensor(points):
    return torch.sigmoid(points[:18])  # Example: First 18 elements of a transformation

# Define optimization parameters
learning_rate = 0.01
max_iter = 10000
tolerance = 1e-6

# Optimization loop
optimizer = torch.optim.SGD([baselineTensor], lr=learning_rate)

for iteration in range(max_iter):
    optimizer.zero_grad()
    
    # Compute tensor from points
    tensor = model(baselineTensor)
    
    # Compute loss
    loss = torch.sum((tensor - target_value) ** 2)
    
    # Check convergence
    if loss.item() < tolerance:
        print(f"Converged at iteration {iteration}")
        break
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()

# Output adjusted points
print("Adjusted points:", baselineTensor.detach().numpy())
FinalPolicy = model(baselineTensor.detach()).detach()
print("tensor", FinalPolicy.numpy())
# np.savetxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/BackgroundOptimizedSimplyFromZero.txt', baselineTensor.detach().numpy())

xaxis = np.arange(18)
plt.bar(xaxis, FinalPolicy.numpy()[0,:])
plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/backgroundOptimized.png', dpi = 1200)
plt.show()
