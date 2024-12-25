import numpy as np
import cv2

# Read the image
image = cv2.imread("/data2/mainul/DataAndGraph/ScoreGettingbetterACERFull.png")

# Apply Gaussian blur
# (5, 5) is the kernel size, and 0 is the standard deviation
blurred_image = cv2.GaussianBlur(image, (5, 5), 1)

# Save or display the result
cv2.imwrite("/data2/mainul/ExplainableAIResults/ScoreGettingbetterACERFullBlurr.png", blurred_image)
