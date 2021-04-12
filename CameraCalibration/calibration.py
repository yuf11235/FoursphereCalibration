#创建者： Zephyr Yan
#时间： 2018.3.14
#功能： 相机标定，得到内外参数，参数的具体含义及转化有待研究

import numpy as np  
import cv2
import glob 


# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001  
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)  
#棋盘格模板规格
w = 4
h = 5
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(2,3,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)  

objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)*4.8 #3为方格实际坐标，单位mm
# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []    # 存储3D点向量 
img_points = []    # 存储2D点向量

# images = glob.glob(r".\*.jpg")
images = glob.glob(r".\images\*.bmp")
                                                         #文件名必须英语
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),0) #高斯平滑处理原图像降噪
    
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
        
        if corners2.any():
            img_points.append(corners2)
        else:
            img_points.append(corners)
        
    cv2.drawChessboardCorners(img, (w,h), corners, ret)
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 640, 480)#指定显示的图的大小，去掉就是默认全屏
    cv2.imshow('img', img)
    cv2.waitKey(10)     #参数为时间，单位MS，是图像停留的时间

cv2.destroyAllWindows()

#==========标定=====
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,size, None, None)
print ("ret:",ret)  
print ("mtx:\n",mtx)        # 内参数矩阵  
print ("dist:\n",dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#k1,k2,k3,k4,k5和k6为径向畸变，p1和p2为轴向畸变。在opencv中，畸变矩阵的参数为(k1,k2,p1,p2[,k3[,k4,k5,k6]]])
print ("旋转向量:\n",rvecs)    # 旋转向量  # 外参数
r = cv2.Rodrigues(np.array(rvecs[0]))[0]    #旋转矩阵
print('旋转矩阵：\n',r)
print ("tvecs:\n",tvecs)    # 平移向量
np.savez("cameraParams.npz",M = mtx,R = r,T = tvecs[0])


