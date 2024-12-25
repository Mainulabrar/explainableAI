import numpy as np

# # Y = np.load("/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/0tpptuning120499.npz")
# Y = np.load("/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/014tpptuning120499.npz")

# # print(Y['l1'].nonzero()[0].size)
# # print(Y['l'+str(10)])

# print(Y['l10'])
# print(Y['l11'])

test_set = ['014', '015', '023', '073', '098']

for patient in test_set:
	Y = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
	repSize = Y['l1'].nonzero()[0].size

	for key in Y.keys():
	    print(f"Key: {key}")
	    print(f"Value:\n{Y[key]}\n")