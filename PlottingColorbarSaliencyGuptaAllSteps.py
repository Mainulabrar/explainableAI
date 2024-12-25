import numpy as np
import matplotlib.pyplot as plt





# Y = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/0xDVHY0step0.npy")
# Yf = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/0xDVHYfull0step0.npy")
# color = np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/NormDifferenceAll.npy')
# color = np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/SfListAll.npy')
color = np.load('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/ValueNormDifferenceAll.npy')

print('colorsize', color.shape) 

for i in range(color.shape[0]):

	Y = np.load(f"/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0xDVHY{i}step0.npy")
	if i != (color.shape[0]-1):
		action = np.load(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0action{i}.npy').item()
		print('action', action)
	# if i == 0:
	# 	print(color[i])

	ColorPTV = color[i,0: 100]
	ColorBladder = color[i, 100: 200]
	ColorRectum = color[i, 200: 300]

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

	# Plot the three datasets against the array of points
	if i != (color.shape[0]-1):
		plt.title(f'{actionDictionary[action]}')
	plt.scatter(x_ptv, y_ptv, c = ColorPTV, vmin=vmin, vmax=vmax, cmap = 'coolwarm', label = 'PTV')
	plt.scatter(x_bladder, y_bladder, c =  ColorBladder, vmin=vmin, vmax=vmax, cmap = 'coolwarm', label='Bladder')
	plt.scatter(x_rectum, y_rectum, c = ColorRectum, vmin=vmin, vmax=vmax, cmap = 'coolwarm', label='Rectum')
	plt.xlim(0, 1.2)
	plt.ylim(0, 1.1)


	# Add annotations pointing to each group of points
	plt.annotate('PTV', xy=(x_ptv.mean(), y_ptv.mean()), xytext=(x_ptv.mean() + 0.1, y_ptv.mean() + 0.1),
	             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
	plt.annotate('Bladder', xy=(x_bladder.mean(), y_bladder.mean()), xytext=(x_bladder.mean() - 0.2, y_bladder.mean() + 0.1),
	             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
	plt.annotate('Rectum', xy=(x_rectum.mean(), y_rectum.mean()), xytext=(x_rectum.mean() + 0.2, y_rectum.mean() - 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

	plt.colorbar()

	plt.savefig(f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}.png', dpi= 1200)
	# plt.savefig(f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0focusedSaliency{i}.png', dpi= 1200)
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