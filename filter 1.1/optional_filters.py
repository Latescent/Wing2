import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = '3.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 5)
cv2.imwrite('gaussian_blur.jpg', gaussian_blur)  # Save Gaussian Blur result

# Apply Median Blur
median_blur = cv2.medianBlur(image, 9)
cv2.imwrite('median_blur.jpg', median_blur)  # Save Median Blur result

# Apply Bilateral Filter
bilateral_blur = cv2.bilateralFilter(image, 10, 75, 75)
cv2.imwrite('bilateral_blur.jpg', bilateral_blur)  # Save Bilateral Filter result

# Apply Non-Local Means Denoising
nlm_denoised = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
cv2.imwrite('nlm_denoised.jpg', nlm_denoised)  # Save Non-Local Means Denoising result