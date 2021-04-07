'''
Created on 2018.3.20
@author: Zephyr Yan
Telephone: 18811346093
Function：灰度重心法提取高光点（批量）
Bug: region.coords#得到的是按行列排列的[i,j]的点坐标
Vision: 2.0
'''
import cv2
import numpy as np
from skimage import data,segmentation,measure,morphology,color
import glob

images = glob.glob("./*.jpg")
num_img = len(images)
Coord_highlight = np.zeros((4,2,num_img))
CoordInWorld = np.zeros((4,2,num_img))
num_img = 0
for fname in images:
    img = cv2.imread(fname) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    gray=cv2.GaussianBlur(gray, (9, 9), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))#圆结构元素
    img = cv2.erode(img,kernel,iterations = 1)
    img = cv2.dilate(img,kernel,iterations = 1)

    retval, img1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)#阈值的选择非常重要

    label_image =measure.label(img1)  #连通区域标记
    #https://www.cnblogs.com/denny402/p/5166258.html
    num_highlight = 0

    for region in measure.regionprops(label_image):
        points = region.coords#得到的是按行列排列的[i,j]
        m,n = points.shape
        x = 0
        y = 0
        g_total = 0
        for i in range(m):
            x = x + gray[points[i,0],points[i,1]]*points[i,1]#X是列项
            y = y + gray[points[i,0],points[i,1]]*points[i,0]#Y是行项
            g_total = g_total + gray[points[i,0],points[i,1]]
        x_focus = x/g_total
        y_focus = y/g_total
        print(x_focus,y_focus)
        #====索引格式为num_highlight：第几个圆；0,1表示x,y；num_img：第几幅图
        Coord_highlight[num_highlight,0,num_img] = x_focus
        Coord_highlight[num_highlight,1,num_img] = y_focus
        num_highlight += 1
    num_img += 1
#===存储为npz格式===
np.savez("./highlightpx/Coord_highlight",Coord_highlight = Coord_highlight)
