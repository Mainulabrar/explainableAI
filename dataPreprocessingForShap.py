import torch
import numpy as np
import re
import os

hidden_size = 32

def state_to_tensor(state):
  	return torch.from_numpy(state).float().unsqueeze(0)


def backgroundForShap():
	ShapData = []
	hx = torch.zeros(1, hidden_size)
	cx = torch.zeros(1, hidden_size)
	for i in range(18):
		pattern = rf'^0xDVHY({i})step0\.npy$'
		for filename in os.listdir('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'):
			matches = re.match(pattern, filename)
			if matches:
				# fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
				fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/',filename)
				# state = state_to_tensor(np.load(fullname))
				ShapData.append(state)

	ShapData = np.array(ShapData)
	ShapData = state_to_tensor(ShapData)
	print('ShapData', ShapData)

	hxBackground = hx.repeat(18,1)
	cxBackground = cx.repeat(18,1)

	background = (ShapData, (hxBackground,cxBackground))

	return background

background = backgroundForShap()
print(background)

# hx = torch.zeros(1, hidden_size)
# print(hx)

# h1x = hx.repeat(2, 1)

# print(h1x)