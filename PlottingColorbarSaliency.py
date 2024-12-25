import numpy as np
import matplotlib.pyplot as plt





Y = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/0xDVHY0step0.npy")
Yf = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/0xDVHYfull0step0.npy")
color = np.load("/data2/mainul/ExplainableAIResults/dataWithPlanscoreRun/NormDifference.npy")

ColorPTV = color[0: 100]
ColorBladder = color[100: 200]
ColorRectum = color[200: 300]

# #next block is for acer==========================================================
print(Y.shape)
print(Y)
Y = np.reshape(Y, (100, 3), order='F')
print(Y)
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

# Plot the three datasets against the array of points
plt.scatter(x_ptv, y_ptv, c = ColorPTV, cmap = 'coolwarm', label = 'PTV Without Perturbation')
plt.scatter(x_bladder, y_bladder, c =  ColorBladder, cmap = 'coolwarm', label='Bladder Without Perturbation')
plt.scatter(x_rectum, y_rectum, c = ColorRectum, cmap = 'coolwarm', label='Rectum Without Perturbation')
plt.xlim(0, 1.2)
plt.ylim(0, 1.1)

plt.colorbar()
plt.show()
plt.close()

# Extract columns to retrieve original variables
y_ptv = Yf[:, 6]
y_bladder = Yf[:, 7]
y_rectum = Yf[:, 8]

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


x_ptv = Yf[:, 9]
x_bladder = Yf[:, 10]
x_rectum = Yf[:, 11]

# x_ptv = x_ptv*100
# x_bladder = x_bladder*100
# x_rectum = x_rectum*100


# x_ptv = np.linspace(0, max(y_ptv), 100)
# x_bladder = np.linspace(0, max(y_bladder), 100)
# x_rectum = np.linspace(0, max(y_rectum), 100)

# Create a plot
plt.figure(figsize=(10, 9))

# Plot the three datasets against the array of points
plt.plot(x_ptv, y_ptv, label='PTV1', color='red', linestyle='-')
plt.plot(x_bladder, y_bladder, label='Bla1', color='green', linestyle='-' )
plt.plot(x_rectum, y_rectum, label='Rec1', color='blue', linestyle='-')
# plt.xlim(0, 1.2)
# plt.ylim(0, 1.1)
plt.show()