'''
Created on 2018.3.29
@author: Zephyr Yan
Telephone: 18811346093
Function：通过相机标定参数，将像素坐标转化为世界坐标，2D to 2D
Improve: 利用矩阵思想，减少for循环，减少了运行时间
Vision: 2.0
'''
import numpy as np  
import cv2
from numpy.linalg import pinv,inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.loadtxt("./Light.txt")
print(a)
x, y, z = a[0,:], a[1,:], a[2,:]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x, y, z, c='y')  # 绘制数据点


ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

