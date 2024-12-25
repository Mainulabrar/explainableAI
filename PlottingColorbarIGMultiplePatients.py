import numpy as np
import matplotlib.pyplot as plt





# Y = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/0xDVHY0step0.npy")
# Yf = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/0xDVHYfull0step0.npy")


# test_set = ['014', '015', '023', '073', '098']
test_set = ['015', '098']

for patient in test_set:
    StepNumIndicator = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
    repSize = StepNumIndicator['l1'].nonzero()[0].size

    for i in range(repSize-1):
	    Y = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}xDVHY120499step{i}.npy")
	    # if i != (repSize-1):
		# actionTaken = np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/0action{i}.npy').item()
	    actionPolicy = np.argmax(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy'))
	    actionPolicyPolicyvalue = np.max(np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy'))
	    Policy = np.load(f"/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/{patient}Policy120499step{i}.npy")
	    PolicySort = np.sort(Policy, axis = 1)[:, ::-1]
        
	    actionPolicy2nd = np.argwhere(Policy == PolicySort[0,1])[0,1].item()
	    actionPolicyPolicyvalue2nd = PolicySort[0,1]
	    # actionTakenPolicyvalue = np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/0policy{i}0.npy')[0,actionTaken]
	    # action = actionPolicy
	    # action = np.array([actionTaken, actionPolicy])
	    action = actionPolicy
	    # action = actionPolicy2nd
	    # IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionRep{i}.txt')
	    # IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionFinalStateBaselineRep{i}.txt')
	    # IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyFinalStateBaselineRep{i}.txt')
	    # IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyTrivialBaselineRep{i}.txt')
	    # IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyOptimizedFromTrivialBaselineRep{i}.txt')
	    # IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attributionTotalPolicyAveOptimizedFromTrivialBaselineRep{i}.txt')
	    IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attribution2ndRep{i}.txt')
	    
	    print('action', action)
		    		# if i == 0:
		# 	print(color[i])
		# IGvalues = np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/0IGvaluestep{i}.npy')
	    SimpleGradSaliency = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}SimpleGradSaliencyStep{i}.txt') 
	    color = SimpleGradSaliency
	    # color = IGvalues

	    ColorPTV = color[0: 100]
	    ColorBladder = color[100: 200]
	    ColorRectum = color[200: 300]

	    actionValues = np.arange(18)
	    actionStrings = [r"$inc\_t_{PTV}$", r"$dec\_t_{PTV}$", r"$inc\_t_{BLA}$", r"$dec\_t_{BLA}$", r"$inc\_t_{REC}$", r"$dec\_t_{REC}$", r'$inc\_\lambda_{PTV}$', r'$dec\_\lambda_{PTV}$', r'$inc\_\lambda_{BLA}$', r'$dec\_\lambda_{BLA}$', r'$inc\_\lambda_{REC}$', r'$dec\_\lambda_{REC}$', r'$inc\_V_{PTV}$', r'$dec\_V_{PTV}$', r'$inc\_V_{BLA}$', r'$dec\_V_{BLA}$', r'$inc\_V_{REC}$', r'$dec\_V_{REC}$']

	    actionDictionary = {value: label for value, label in zip(actionValues, actionStrings)}

	    print(ColorPTV)
	    print(ColorBladder)
	    print(ColorRectum)
        # #next block is for acer==========================================================
	    print(Y.shape)
		# print(Y)
	    Y = np.reshape(Y, (100, 3), order='F')
		# print(Y)
		# #=================================================================================

		# Extract columns to retrieve original variables
	    y_ptv = Y[:, 0]
	    y_bladder = Y[:, 1]
	    y_rectum = Y[:, 2]

		# y_ptv = y_ptv/max(y_ptv)
		# y_bladder = y_bladder/max(y_bladder)
		# y_rectum = y_rectum/max(y_rectum)

		# Trial
	    x_ptv = np.linspace(1, 1.15, 100)
	    x_bladder = np.linspace(0.6, 1.1, 100)
	    x_rectum = np.linspace(0.6, 1.1, 100)

		# x_ptv = np.linspace(0, max(y_ptv), 100)
		# x_bladder = np.linspace(0, max(y_bladder), 100)
		# x_rectum = np.linspace(0, max(y_rectum), 100)

		# Create a plot
	    plt.figure(figsize=(10, 6))

	    vmin = min(ColorPTV.min(), ColorBladder.min(), ColorRectum.min())
	    vmax = max(ColorPTV.max(), ColorBladder.max(), ColorRectum.max())

	    x_positions = [0.755, 0.812, 0.881, 0.943, 1.006, 1.1]
	    y_positions = [0.2, 0.3, 0.4, 0.55]

	    for x_pos in x_positions:
	        plt.axvline(x=x_pos, color='black', linestyle='-', alpha=0.7)

	    for y_pos in y_positions:
	        plt.axhline(y=y_pos, color='black', linestyle='-', alpha=0.7)


		# values = value = np.load(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/0ValueRep{i}step0.npy')
		# allrewards = np.loadtxt('/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/dataWithPlanscoreRun/0Allrewards.txt')
		# Plot the three datasets against the array of points
	    # if i != (repSize-1):
		# plt.title(f'rep:{i} actionTaken{actionDictionary[action[0]]} Value{values[0,action[0]]:.4f} reward{allrewards[i]:.4f} policy{actionTakenPolicyvalue:.4f},\nactionPolicy{actionDictionary[action[1]]} Value{values[0,action[1]]:.4f} policy{actionPolicyPolicyvalue:.4f}')
	    plt.title(f'Patient {patient} step:{i} actionPolicy{actionDictionary[action]} policy{actionPolicyPolicyvalue:.4f}')
	    # plt.title(f'Patient {patient} step:{i} actionPolicy{actionDictionary[action]} policy{actionPolicyPolicyvalue2nd:.4f}')
	    
	    plt.scatter(x_ptv, y_ptv, c = ColorPTV, vmin=vmin, vmax=vmax, cmap = 'coolwarm', label = 'PTV')
	    plt.scatter(x_bladder, y_bladder, c =  ColorBladder, vmin=vmin, vmax=vmax, cmap = 'coolwarm', label='Bladder')
	    plt.scatter(x_rectum, y_rectum, c = ColorRectum, vmin=vmin, vmax=vmax, cmap = 'coolwarm', label='Rectum')
	    plt.xlim(0.6, 1.2)
	    plt.ylim(0, 1.1)



		# Find the first point from the left for each scatter plot
	    ptv_first_point = (x_ptv.min(), y_ptv[x_ptv.argmin()])  # Coordinates of the first point from the left
	    bladder_first_point = (x_bladder.min(), y_bladder[x_bladder.argmin()])  # Coordinates of the first point from the left
	    rectum_first_point = (x_rectum.min(), y_rectum[x_rectum.argmin()])  # Coordinates of the first point from the left

		# Update the annotations
	    plt.annotate(
		    'PTV', xy=ptv_first_point, 
		    xytext=(ptv_first_point[0] + 0.1, ptv_first_point[1] + 0.1),
		    arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA = 5, shrinkB=5, mutation_scale=10), fontsize=10
	    )
	    plt.annotate(
		    'Bladder', xy=bladder_first_point, 
		    xytext=(bladder_first_point[0] - 0.2, bladder_first_point[1] + 0.1),
		    arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA = 5, shrinkB=5, mutation_scale=10), fontsize=10
	    )
	    plt.annotate(
		    'Rectum', xy=rectum_first_point, 
		    xytext=(rectum_first_point[0] + 0.2, rectum_first_point[1] - 0.1),
		    arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA = 5, shrinkB=5, mutation_scale=10), fontsize=10
	    )



			# # Add annotations pointing to each group of points
			# plt.annotate('PTV', xy=(x_ptv.mean(), y_ptv.mean()), xytext=(x_ptv.mean() + 0.1, y_ptv.mean() + 0.1),
			#              arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
			# plt.annotate('Bladder', xy=(x_bladder.mean(), y_bladder.mean()), xytext=(x_bladder.mean() - 0.2, y_bladder.mean() + 0.1),
			#              arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
			# plt.annotate('Rectum', xy=(x_rectum.mean(), y_rectum.mean()), xytext=(x_rectum.mean() + 0.2, y_rectum.mean() - 0.1),
			#          arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

	    plt.colorbar()

	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraph{i}.png', dpi= 1200)
	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesFinalStateBaselineGraph{i}.png', dpi= 1200)
	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesTotalTensorFinalStateBaselineGraph{i}.png', dpi= 1200)
	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesTotalTensorTrivialZeroBaselineGraph{i}.png', dpi= 1200)
	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesTotalTensorOptimizedFromZeroBaselineGraph{i}.png', dpi= 1200)	    
	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesTotalTensorAveOptimizedFromZeroBaselineGraph{i}.png', dpi= 1200)	    
	    # plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValues2ndActionZeroBase{i}.png', dpi= 1200)	    
	    plt.savefig(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}SimpleGradSaliencyStep{i}.png', dpi= 1200)	    	    
	    # plt.show()
	    plt.close()

# # Extract columns to retrieve original variables
# y_ptv = Yf[:, 6]
# y_bladder = Yf[:, 7]
# y_rectum = Yf[:, 8]

# y_ptv = y_ptv*100
# y_bladder = y_bladder*100
# y_rectum = y_rectum*100


# y_ptv = y_ptv/max(y_ptv)
# y_bladder = y_bladder/max(y_bladder)
# y_rectum = y_rectum/max(y_rectum)

# Trial
# x_ptv = np.linspace(1, 1.15, 100)
# x_bladder = np.linspace(0.6, 1.1, 100)
# x_rectum = np.linspace(0.6, 1.1, 100)


# x_ptv = Yf[:, 9]
# x_bladder = Yf[:, 10]
# x_rectum = Yf[:, 11]

# # x_ptv = x_ptv*100
# # x_bladder = x_bladder*100
# # x_rectum = x_rectum*100


# # x_ptv = np.linspace(0, max(y_ptv), 100)
# # x_bladder = np.linspace(0, max(y_bladder), 100)
# # x_rectum = np.linspace(0, max(y_rectum), 100)

# # Create a plot
# plt.figure(figsize=(10, 9))

# # Plot the three datasets against the array of points
# plt.plot(x_ptv, y_ptv, label='PTV1', color='red', linestyle='-')
# plt.plot(x_bladder, y_bladder, label='Bla1', color='green', linestyle='-' )
# plt.plot(x_rectum, y_rectum, label='Rec1', color='blue', linestyle='-')
# # plt.xlim(0, 1.2)
# # plt.ylim(0, 1.1)
# plt.show()