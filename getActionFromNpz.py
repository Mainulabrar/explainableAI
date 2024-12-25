import numpy as np

# Y = np.load("/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/0tpptuning120499.npz")


# repSize = Y['l1'].nonzero()[0].size
# print(repSize)


paraMax = 100000  # change in validation as well
paraMin = 0.1
paraMax_tPTV = 1.2
paraMin_tPTV = 1
paraMax_tOAR = 1
paraMax_VOAR = 1
paraMax_VPTV = 0.3

def determine_action(prev_tPTV, curr_tPTV, prev_tBLA, curr_tBLA, prev_tREC, curr_tREC, prev_lambdaPTV, curr_lambdaPTV, prev_lambdaBLA, curr_lambdaBLA, prev_lambdaREC, curr_lambdaREC, prev_VPTV, curr_VPTV, prev_VBLA, curr_VBLA, prev_VREC, curr_VREC):


	paraMax = 100000  # change in validation as well
	paraMin = 0.1
	paraMax_tPTV = 1.2
	paraMin_tPTV = 1
	paraMax_tOAR = 1
	paraMax_VOAR = 1
	paraMax_VPTV = 0.3

	if prev_tPTV != paraMax_tPTV and curr_tPTV == min(prev_tPTV * 1.01, paraMax_tPTV):
		return 0
	elif prev_tPTV != paraMin_tPTV and curr_tPTV == max(prev_tPTV * 0.91, paraMin_tPTV):
	    return 1
	elif prev_tBLA != paraMax_tOAR and curr_tBLA == min(prev_tBLA * 1.25, paraMax_tOAR):
	    return 2
	elif curr_tBLA == prev_tBLA * 0.6:
	    return 3
	elif prev_tREC != paraMax_tOAR and curr_tREC == min(prev_tREC * 1.25, paraMax_tOAR):
	    return 4
	elif curr_tREC == prev_tREC * 0.6:
	    return 5
	elif curr_lambdaPTV == prev_lambdaPTV * 1.65:
	    return 6
	elif curr_lambdaPTV == prev_lambdaPTV * 0.6:
	    return 7
	elif curr_lambdaBLA == prev_lambdaBLA * 1.65:
	    return 8
	elif curr_lambdaBLA == prev_lambdaBLA * 0.6:
	    return 9
	elif curr_lambdaREC == prev_lambdaREC * 1.65:
	    return 10
	elif curr_lambdaREC == prev_lambdaREC * 0.6:
	    return 11
	elif prev_VPTV != paraMax_VPTV and curr_VPTV == min(prev_VPTV * 1.25, paraMax_VPTV):
	    return 12
	elif curr_VPTV == prev_VPTV * 0.8:
	    return 13
	elif prev_VBLA != paraMax_VOAR and curr_VBLA == min(prev_VBLA * 1.25, paraMax_VOAR):
	    return 14
	elif curr_VBLA == prev_VBLA * 0.8:
	    return 15
	elif prev_VREC != paraMax_VOAR and curr_VREC == min(prev_VREC * 1.25, paraMax_VOAR):
	    return 16
	elif curr_VREC == prev_VREC * 0.8:
	    return 17
	elif prev_tPTV == paraMax_tPTV and prev_tBLA != paraMax_tOAR and prev_tREC != paraMax_tOAR and prev_VPTV != paraMax_VPTV and prev_VBLA != paraMax_VOAR and prev_VREC != paraMax_VOAR:
		return 0 
	elif prev_tPTV == paraMin_tPTV and prev_tBLA != paraMax_tOAR and prev_tREC != paraMax_tOAR and prev_VPTV != paraMax_VPTV and prev_VBLA != paraMax_VOAR and prev_VREC != paraMax_VOAR:
		return 1
	elif prev_tPTV != paraMin_tPTV and prev_tBLA == paraMax_tOAR and prev_tREC != paraMax_tOAR and prev_VPTV != paraMax_VPTV and prev_VBLA != paraMax_VOAR and prev_VREC != paraMax_VOAR:
		return 2
	elif prev_tPTV != paraMin_tPTV and prev_tBLA != paraMax_tOAR and prev_tREC == paraMax_tOAR and prev_VPTV != paraMax_VPTV and prev_VBLA != paraMax_VOAR and prev_VREC != paraMax_VOAR:
		return 4
	elif prev_tPTV != paraMin_tPTV and prev_tBLA != paraMax_tOAR and prev_tREC != paraMax_tOAR and prev_VPTV == paraMax_VPTV and prev_VBLA != paraMax_VOAR and prev_VREC != paraMax_VOAR:
		return 12
	elif prev_tPTV != paraMin_tPTV and prev_tBLA != paraMax_tOAR and prev_tREC != paraMax_tOAR and prev_VPTV != paraMax_VPTV and prev_VBLA == paraMax_VOAR and prev_VREC != paraMax_VOAR:
		return 14
	elif prev_tPTV != paraMin_tPTV and prev_tBLA != paraMax_tOAR and prev_tREC != paraMax_tOAR and prev_VPTV != paraMax_VPTV and prev_VBLA != paraMax_VOAR and prev_VREC == paraMax_VOAR:
		return 16
	else:
	    return None


# for i in range(int(repSize-1)):
# 	action = determine_action(Y['l1'][i], Y['l1'][i+1], Y['l2'][i], Y['l2'][i+1],Y['l3'][i], Y['l3'][i+1],Y['l4'][i], Y['l4'][i+1],Y['l5'][i], Y['l5'][i+1],Y['l6'][i], Y['l6'][i+1],Y['l7'][i], Y['l7'][i+1],Y['l8'][i], Y['l8'][i+1],Y['l9'][i], Y['l9'][i+1])
# 	print(action)
# 	np.save(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0action{i}', action)

test_set = ['014', '015', '023', '073', '098']

for patient in test_set:
	Y = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
	repSize = Y['l1'].nonzero()[0].size

	for key in Y.keys():
	    print(f"Key: {key}")
	    print(f"Value:\n{Y[key]}\n")

	# Close the file after accessing
	# Y.close()
	for i in range(int(repSize-1)):
		action = determine_action(Y['l1'][i], Y['l1'][i+1], Y['l2'][i], Y['l2'][i+1],Y['l3'][i], Y['l3'][i+1],Y['l4'][i], Y['l4'][i+1],Y['l5'][i], Y['l5'][i+1],Y['l6'][i], Y['l6'][i+1],Y['l7'][i], Y['l7'][i+1],Y['l8'][i], Y['l8'][i+1],Y['l9'][i], Y['l9'][i+1])
		print(f'step{i}',action)
		np.save(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}action{i}', action)


