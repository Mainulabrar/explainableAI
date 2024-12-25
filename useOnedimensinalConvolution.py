import numpy as np
from scipy.ndimage import convolve1d


# Parameters
sigma = 1.0
kernel_size = int(6 * sigma + 1)  # 6*sigma gives a wide enough kernel for most uses
x = np.linspace(-3*sigma, 3*sigma, kernel_size)

# 1D Gaussian kernel
gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel



# Your 1D signal or data
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Convolve the signal with the Gaussian kernel
smoothed_signal = convolve1d(signal, gaussian_kernel)

print(smoothed_signal)