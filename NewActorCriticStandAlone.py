import torch
import os
import re
import numpy as np
from torch import nn
from torch.nn import functional as F
from gym.spaces import Discrete

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super(ActorCritic, self).__init__()
        self.state_size = 300
        self.action_size = action_space.n

        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, self.action_size)
        self.fc_critic = nn.Linear(hidden_size, self.action_size)

    def forward(self, x, h=None):
        if h is None:
            batch_size = x.size(0)
            hx = torch.zeros(batch_size, 32, device=x.device)
            cx = torch.zeros(batch_size, 32, device=x.device)
            h = (hx, cx)

        x = F.relu(self.fc1(x))
        h = self.lstm(x, h)
        x = h[0]
        policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)
        return policy

def backgroundForShap():
    ShapData = []
    hx = torch.zeros(1, 32)
    cx = torch.zeros(1, 32)
    for i in range(18):
        pattern = rf'^0xDVHY({i})step0\.npy$'
        for filename in os.listdir('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'):
            matches = re.match(pattern, filename)
            if matches:
                fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/', filename)
                ShapData.append(np.load(fullname))

    ShapData = np.array(ShapData).reshape(-1, 300)
    ShapData = torch.from_numpy(ShapData).float()

    hxBackground = hx.repeat(ShapData.size(0), 1)
    cxBackground = cx.repeat(ShapData.size(0), 1)
    hBackground = (hxBackground.float(), cxBackground.float())

    print("ShapData shape:", ShapData.shape)
    print("hxBackground shape:", hxBackground.shape)
    print("cxBackground shape:", cxBackground.shape)

    return (ShapData, hBackground)

# Initialize model
model = ActorCritic(np.zeros([300]), Discrete(18), 32)
model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
model.eval()

# Create background data
background = backgroundForShap()
output = model(*background)

print(output)
