'''
Created on 2018.3.20
@author: Zephyr Yan
Telephone: 18811346093
Function：canny算子得到圆边缘信息，HoughCircles自动找圆，最后根据一定的范围大小
         找到边缘像素坐标并存储
Vision: 2.0
Improve：解决了上一版本不能自动找到图中所有圆信息的问题
'''  
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

       
img = cv2.imread("./3.jpg")  #Canny只能处理灰度图，所以将读取的图像转成灰度图  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
img1 = cv2.GaussianBlur(gray,(9,9),0) #高斯平滑处理原图像降噪
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))#圆结构元素
img1 = cv2.erode(img1,kernel,iterations = 1)
img1 = cv2.dilate(img1,kernel,iterations = 1)
canny = cv2.Canny(img1, 45, 100)

cv2.namedWindow("Canny",0)
cv2.resizeWindow("Canny", 640, 480)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT,1,100,param1=100,
                            param2=10,minRadius=250,maxRadius=300)
print(circles)
circles = circles[0,:]
Ball_num = len(circles)
outline_px = []
for B_num in range(Ball_num):
    #cv2.circle(img,(circles[i,0],circles[i,1]),circles[i,2],(255,0,0),5)  
    #cv2.circle(img,(circles[i,0],circles[i,1]),2,(255,0,255),10)  
    cv2.rectangle(gray,
                  (int(circles[B_num,0]-circles[B_num,2]-50),
                   int(circles[B_num,1]-circles[B_num,2]-50)),
                  (int(circles[B_num,0]+circles[B_num,2]+50),
                   int(circles[B_num,1]+circles[B_num,2]+50)),
                  (255,0,0),5) 
    print("圆心坐标",circles[B_num,0],circles[B_num,1])
    x = []
    y = []
    N = 0
    for j in range(int(circles[B_num,0]-circles[B_num,2]-50),
                   int(circles[B_num,0]+circles[B_num,2]+50)):
        for i in range(int(circles[B_num,1]-circles[B_num,2]-50),
                       int(circles[B_num,1]+circles[B_num,2]+50)):
            if canny[i,j] == 255:
                x.append(j)
                y.append(i)
                N+=1
                cv2.circle(gray,(j,i),2,(255,0,0),2)
    outline_Bnum = np.zeros((N,2))
    outline_Bnum[:,0] = x
    outline_Bnum[:,1] = y
    outline_px.append(outline_Bnum)
np.savez("./circle_outline.npz",px=outline_px)
print(outline_px)
plt.figure(12)
plt.subplot(221),plt.imshow(gray)
plt.subplot(222),plt.imshow(canny)  
plt.subplot(212)
for L_num in range(Ball_num):
    plt.scatter(outline_px[L_num][:,0],outline_px[L_num][:,1],
                color="red") #画样本点
ax = plt.gca()  #获取到当前坐标轴信息
ax.xaxis.set_ticks_position('top')   #将X坐标轴移到上面
ax.invert_yaxis()   
plt.show()
