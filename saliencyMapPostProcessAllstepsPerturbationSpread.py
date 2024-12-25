import numpy as np
import re
import os


pattern = r'^(\d+)policy(\d+)0\.npy$'

# for filename in os.listdir('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'):
# 	matches = re.match(pattern, filename)
# 	if matches:

# 		fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
# 		print(np.load(fullname))
# 		print(matches.group(1))

list1 = []


SizeIndicator = np.load("/data2/mainul1/results_CORS/scratch6_30StepsNewParamenters3indCriTime/dataWithPlanscoreRun/0tpptuning120499.npz")

repSize = SizeIndicator['l1'].nonzero()[0].size
spreadArray = np.zeros((int(repSize), 301, 300))

for rep in range(int(repSize)):
	print('rep', rep)
	for i in range(301):
		pattern = rf'^({i})xDVHY({rep})step0\.npy$'
		# for filename in os.listdir('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'):
		for filename in os.listdir('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'):
			matches = re.match(pattern, filename)
			if matches:
				# fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
				fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/',filename)
				# print(np.load(fullname))
				print(matches.group(1))
				list1.append(matches.group(1))
				# DVHArray = np.load(fullname)
				spreadArray[rep, i, :] = np.load(fullname)






# print(list1)
print(spreadArray)

euNormAll = []
for rep in range(int(repSize)):
	euNorm = []
	for i in range(300):
		distance = np.linalg.norm(spreadArray[rep, 0, i] - spreadArray[rep, i+1, i])
		euNorm.append(distance)
	euNormAll.append(euNorm)

print(euNormAll)

# np.save('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'+'NormDifference', euNorm)
np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'perturbationSpreadAll', euNormAll)