import numpy as np
import re
import os

# The following is the first part of the data process======================================

# pattern = r'^(\d+)policy(\d+)0\.npy$'

# # for filename in os.listdir('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'):
# # 	matches = re.match(pattern, filename)
# # 	if matches:

# # 		fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
# # 		print(np.load(fullname))
# # 		print(matches.group(1))

# list1 = []
# listValue = []
# listQvalue = []
# actionArray = []

SizeIndicator = np.load("/data2/mainul1/results_CORS/scratch6_30StepsNewParamenters3indCriTime/dataWithPlanscoreRun/0tpptuning120499.npz")

repSize = SizeIndicator['l1'].nonzero()[0].size
# policyArray = np.zeros((int(repSize), 301, 18))
# valueArray = np.zeros((int(repSize), 301))
# QvalueArray = np.zeros((int(repSize), 301, 18))


# for rep in range(int(repSize)):
# 	print('rep', rep)
# 	for i in range(301):
# 		pattern = rf'^({i})policy({rep})0\.npy$'
# 		patternValue = rf'^({i})value({rep})0\.npy$'
# 		patternQvalue = rf'^({i})Qvalue({rep})0\.npy$'
# 		# for filename in os.listdir('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'):
# 		for filename in os.listdir('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'):
# 			# matches = re.match(pattern, filename)
# 			matchesValue = re.match(patternValue, filename)
# 			matchesQvalue = re.match(patternQvalue, filename)
# 			# if matches:
# 			# 	# fullname = os.path.join('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/',filename)
# 			# 	fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/',filename)
# 			# 	# print(np.load(fullname))
# 			# 	print(matches.group(1))
# 			# 	list1.append(matches.group(1))
# 			# 	policyArray[rep, i, :] = np.load(fullname)

# 			if matchesValue:
# 				fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/',filename)
# 				# print(np.load(fullname))
# 				print(matchesValue.group(1))
# 				listValue.append(matchesValue.group(1))
# 				valueArray[rep, i] = np.load(fullname)
# 			elif matchesQvalue:
# 				fullname = os.path.join('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/',filename)
# 				# print(np.load(fullname))
# 				print(matchesQvalue.group(1))
# 				listQvalue.append(matchesQvalue.group(1))
# 				QvalueArray[rep, i, :] = np.load(fullname)

# 	actionArray.append(np.max(policyArray[rep, 0, :]))


# np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'AllPolicyArray', policyArray)
# np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'AllValueArray', valueArray)
# np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'AllQvalueArray', QvalueArray)

#=================================================================================

QvalueArray = np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/AllQvalueArray.npy')
valueArray = np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/AllValueArray.npy')
policyArray = np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/AllPolicyArray.npy')

actionArray = []
for rep in range(int(repSize)):
	actionArray.append(np.max(policyArray[rep, 0, :]))

actionArray = np.array(actionArray)

np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'actionArray', actionArray)

def kl_divergence(P, Q):
    """
    Calculate the Kullback-Leibler divergence D_KL(P || Q)
    :param P: First probability distribution (array-like)
    :param Q: Second probability distribution (array-like)
    :return: KL divergence (float)
    """
    # Ensure P and Q are NumPy arrays
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)

    # Add a small epsilon to avoid log(0) or division by zero
    epsilon = 1e-10
    P = np.clip(P, epsilon, 1)
    Q = np.clip(Q, epsilon, 1)

    # Calculate KL divergence
    return np.sum(P * np.log(P / Q))

deltaPlistAll =[]
DKLlistAll = []
SfListAll = []

for rep in range(int(repSize)):
	# euNorm = []
	deltaPlist = []
	DKLlist = []
	SfList = []
	Pinit = np.exp(QvalueArray[rep, 0, int(actionArray[rep])])/np.sum(np.exp(QvalueArray[rep, 0, :]))
	QremInit = np.delete(QvalueArray[rep, 0, :], int(actionArray[rep]))
	PremInit = np.exp(QremInit)/np.sum(np.exp(QremInit))
	for i in range(300):
		P = np.exp(QvalueArray[rep, i, int(actionArray[rep])])/np.sum(np.exp(QvalueArray[rep, i, :]))
		deltaP = Pinit - P
		Qrem = np.delete(QvalueArray[rep, i, :], int(actionArray[rep]))
		Prem = np.exp(Qrem)/np.sum(np.exp(Qrem))
		DKL = kl_divergence(Prem, PremInit)
		K = 1/(1+DKL)
		Sf = (2*K*deltaP)/(K+deltaP)
		deltaPlist.append(deltaP)
		DKLlist.append(DKL)
		SfList.append(Sf)

	deltaPlistAll.append(deltaPlist)
	DKLlistAll.append(DKLlist)
	SfListAll.append(SfList)

np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'deltaPlistAll', deltaPlistAll)
np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'DKLlistAll', DKLlistAll)
np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'SfListAll', SfListAll)

# the following part is for value only saliency map ================================================
euNormAll = []
for rep in range(int(repSize)):
	euNorm = []
	for i in range(300):
		distance = np.linalg.norm(valueArray[rep, 0] - valueArray[rep, i+1])
		euNorm.append(distance)
	euNormAll.append(euNorm)

# print(euNormAll)


# np.save('/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/'+'NormDifference', euNorm)
np.save('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/'+'ValueNormDifferenceAll', euNormAll)