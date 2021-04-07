'''
Created on 2018.3.15
@author: Zephyr Yan
Telephone: 18811346093
Function：提取轮廓线及高光点
Bug: 
'''
import numpy as np
import cv2


img = cv2.imread("11.jpg",0)
img = cv2.GaussianBlur(img,(9,9),0) #高斯平滑处理原图像降噪
retval, img1 = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)#阈值需要选择正确，对结果影响大

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))#圆结构元素
img = cv2.erode(img,kernel,iterations = 1)
img = cv2.dilate(img,kernel,iterations = 1)
outline_img = cv2.Canny(img, 25, 50)

m,n = img.shape
for i in range(m):
    for j in range(n):
        if img1[i,j] == 255:
            outline_img[i,j] = 255

cv2.namedWindow("Hlight",0)
cv2.resizeWindow("Hlight", 640, 480)
cv2.imshow("Hlight",outline_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
