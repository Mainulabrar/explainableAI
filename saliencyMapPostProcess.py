import numpy as np
import re
import os


pattern = r'^(\d+)policy00\.npy$'

# for filename in os.listdir('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'):
# 	matches = re.match(pattern, filename)
# 	if matches:

# 		fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
# 		print(np.load(fullname))
# 		print(matches.group(1))

list1 = []

policyArray = np.zeros((301, 18))

for i in range(301):
	pattern = rf'^({i})policy00\.npy$'
	for filename in os.listdir('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'):
		matches = re.match(pattern, filename)
		if matches:
			fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
			# print(np.load(fullname))
			print(matches.group(1))
			list1.append(matches.group(1))
			policyArray[i, :] = np.load(fullname)



# print(list1)
print(policyArray)

euNorm = []

for i in range(300):
	distance = np.linalg.norm(policyArray[0, :] - policyArray[i+1, :])
	euNorm.append(distance)

print(euNorm)

np.save('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'+'NormDifference', euNorm)