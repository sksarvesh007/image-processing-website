import cv2
import numpy as np 
number = 2
img = cv2.imread(f'images/{number}.jpg', 1)
rows , col = img.shape[:2]
if number == 1:
    img = cv2.resize(img, (1280, 720))
elif number == 2:
    img= cv2.resize(img, (720, 720))


#gaussian kernel for sharpening
gaussian_blur=cv2.GaussianBlur(img, (25,25), 0)

#sharpening using addweighted()
sharpened1 = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
sharpened2 = cv2.addWeighted(img, 3.5, gaussian_blur, -2.5, 0)
sharpened3 = cv2.addWeighted(img, 7.5, gaussian_blur, -6.5, 0)



cv2.imwrite(f'sharpening output/{number}/original_{number}.jpg', img)
cv2.imwrite(f'sharpening output/{number}/sharpened1_{number}.jpg', sharpened1)
cv2.imwrite(f'sharpening output/{number}/sharpened2_{number}.jpg', sharpened2)
cv2.imwrite(f'sharpening output/{number}/sharpened3_{number}.jpg', sharpened3)


cv2.imshow('Original', img)
cv2.imshow('Sharpened1', sharpened1)
cv2.imshow('Sharpened2', sharpened2)
cv2.imshow('Sharpened3', sharpened3)

cv2.waitKey(0)