import cv2 
import numpy as np

number = 2
img = cv2.imread(f'images/{number}.jpg', 1)
rows , col = img.shape[:2]
if number == 1:
    img = cv2.resize(img, (1280, 720))
elif number == 2:
    img = cv2.resize(img, (720, 720))

#kernel blurring
kernel = np.ones((25,25), np.float32)/625.0
blurred_img = cv2.filter2D(img, -1, kernel)

#box filter 
output_box_blur = cv2.blur(img, (25,25))
output_box = cv2.boxFilter(img, -1, (5,5) , normalize=False)

#Gaussian filter
#gaussian blur is considered to be the best blur for removing noise
output_gaussian = cv2.GaussianBlur(img, (25,25), 0)

#median blur
output_median = cv2.medianBlur(img, 5)

#bilateral filter (edge preserving + noise removal)
bilateralfilter = cv2.bilateralFilter(img, 9, 75, 75)



cv2.imwrite(f'output/{number}/gaussian_{number}.jpg', output_gaussian)
cv2.imwrite(f'output/{number}/original_{number}.jpg', img)
cv2.imwrite(f'output/{number}/blurred_kernel_{number}.jpg', blurred_img)
cv2.imwrite(f'output/{number}/box_blur_{number}.jpg', output_box_blur)
cv2.imwrite(f'output/{number}/box_{number}.jpg', output_box)
cv2.imwrite(f'output/{number}/median_{number}.jpg', output_median)
cv2.imwrite(f'output/{number}/bilateral_{number}.jpg', bilateralfilter)



cv2.imshow('Original', img) 
cv2.imshow('kernel Blurred', blurred_img)
cv2.imshow('Box Blur', output_box_blur)
cv2.imshow('Box', output_box)
cv2.imshow('Gaussian', output_gaussian)
cv2.imshow('median blur' , output_median)
cv2.imshow('bilateral filter', bilateralfilter)
cv2.waitKey(0)