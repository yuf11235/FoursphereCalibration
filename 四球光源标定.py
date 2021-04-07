'''
Created on 2018.3.24
@author: Zephyr Yan
Telephone: 18811346093
Function：多参考球光源空间位置标定，参考平面为世界坐标系，设图像平面Zw=0
version: 2.1
Improve: 修改之前代码偏斜的错误
'''
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import scipy.optimize
from numpy.linalg import norm, svd, pinv,eig
import Px2World as PTW

up_light = 8
down_light = 24
Light_num = 32 #光源数量
Ball_num = 4 #参考球个数
r = 5 #实际半径，单位：mm

cameraParams = np.load("./cameraParams.npz")
Point0_world = cameraParams["T"]
camera_world = np.array([-Point0_world[0,0],
                         -Point0_world[1,0],
                         Point0_world[2,0]])
#print(camera_world)

outline_world3d =[] #轮廓世界坐标
outline_world2d1 = []
for B_num in range(Ball_num):
    Ball_data = np.load("./circle_outline.npz")
    outline_px = Ball_data["px"][B_num]
    outline_world2d1.append(outline_px)
    point_num = len(outline_px)
    #print(point_num)
    outline_world2d = PTW.pxtoworld(cameraParams,outline_px)
    Z_outline = np.zeros((point_num,1))
    outline_world3d.append(np.hstack((outline_world2d,Z_outline)))

highlight = np.load("./up8/highlightpx/Coord_highlight.npz")
highlight1 = np.load("./down24/highlightpx/Coord_highlight.npz")
PointP_world2d1 = []
PointP_world3d = [] #存储高光点世界坐标
for L_num in range(up_light):
    PointP_px = highlight["Coord_highlight"][:,:,L_num]
    PointP_world2d1.append(PointP_px)
    PointP_world2d = PTW.pxtoworld(cameraParams,PointP_px)
    Z_pointP = np.zeros((len(PointP_world2d),1))
    PointP_world3d.append(np.hstack((PointP_world2d,Z_pointP)))#列表形式存储3d数据
#print(PointP_world3d)
for L_num in range(down_light):
    PointP_px = highlight1["Coord_highlight"][:,:,L_num]
    PointP_world2d1.append(PointP_px)
    PointP_world2d = PTW.pxtoworld(cameraParams,PointP_px)
    Z_pointP = np.zeros((len(PointP_world2d),1))
    PointP_world3d.append(np.hstack((PointP_world2d,Z_pointP)))
    
outline_vec = [] 
for B_num in range(Ball_num):
    point_num = len(outline_world3d[B_num])
    outline_vec1 = np.zeros((point_num,3))
    for p_num in range(point_num):
        outline_vec1[p_num,:] = outline_world3d[B_num][p_num,:] - camera_world
        outline_vec1[p_num,:] = outline_vec1[p_num,:]/norm(
            outline_vec1[p_num,:])#单位化
    outline_vec.append(outline_vec1)
    
Vos = np.zeros((Ball_num,3)) #Vos向量
cosBeta = np.zeros(Ball_num) #
sinBeta = np.zeros(Ball_num)
PointS_world = np.zeros((Ball_num,3)) #球心世界坐标
PointS_world1 = np.zeros((Ball_num,3))
for B_num in range(Ball_num):
    outline_vec_mean = np.mean(outline_vec[B_num],0)
    A_matrix = outline_vec[B_num] - outline_vec_mean
    
    U,S,Vh = svd(A_matrix)
    #print(Vh,"\n",S)
    Vos[B_num,:] = Vh[2,:]
    '''
    #同SVD方法结果差不多，D为奇异值，V为向量
    D,V = eig(A_matrix.T.dot(A_matrix))
    #print(D,"\n",V)
    k = np.argmin(D)#找到最小的奇异值
    Vos1[B_num,:] = V[:,k].T#取列向量为
    '''
    cosBeta1 = np.dot(outline_vec[B_num], Vos[B_num,:])
    cosBeta1 = np.min(cosBeta1)
    #print(cosBeta1)
    
    cosBeta[B_num] = np.dot(outline_vec_mean, Vos[B_num,:])
    #print(cosBeta[B_num])
    if cosBeta[B_num] < 0: #cosBeta应该为正值，Beta理论上是小于90度的
        Vos[B_num,:] = -Vos[B_num,:]
    sinBeta[B_num] = (1-(cosBeta[B_num])**2)**0.5
    PointS_world1[B_num,:]= (r/sinBeta[B_num])*Vos[B_num,:]
    PointS_world[B_num,:] = (r/sinBeta[B_num])*Vos[B_num,:] + camera_world
#print(cosBeta)
print(PointS_world)
'''
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax.scatter(PointS_world[:,0],PointS_world[:,1],PointS_world[:,2], s=100, c='r')
ax.scatter(PointS_world1[:,0],PointS_world1[:,1],PointS_world1[:,2], s=100, c='r')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

#=======将标定好的空间坐标显示在3D图中
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
for L_num in range(Ball_num):
    ax.scatter(outline_world2d1[L_num][:,0],outline_world2d1[L_num][:,1],
                s=100, c='y')
for L_num in range(Light_num):
    ax.scatter(PointP_world2d1[L_num][:,0],PointP_world2d1[L_num][:,1],
                s=100, c='b')
#ax.scatter(PointS_world[:,0],PointS_world[:,1],PointS_world[:,2], s=100, c='r')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
'''
OP_unit = [] #存储单位OP向量，行向量为单位OP向量
for L_num in range(Light_num):
    OP_vec = np.zeros((4,3))
    for B_num in range(Ball_num):
        OP_vec1 = PointP_world3d[L_num][B_num,:] - camera_world
        OP_vec[B_num,:] = OP_vec1/norm(OP_vec1)
    OP_unit.append(OP_vec)
#print(OP_unit)
'''
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
for L_num in range(Light_num):
    ax.scatter(OP_unit[L_num][:,0],OP_unit[L_num][:,1],
               OP_unit[L_num][:,2], s=100, c='y')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
'''
PointP_real = [] #存储
def quadratic(a,b,c):  
    p=b*b-4*a*c  
    if p>=0 and a!=0:#一元二次方程有解的条件  
        x1=(-b+math.sqrt(p))/(2*a)  
        x2=(-b-math.sqrt(p))/(2*a)  
        return x1,x2  
    elif a==0:#a=0的情况下为一元一次方程  
        x1=x2=-c/b  
        return x1  
    else:  
        return('Wrong Number！')
for L_num in range(Light_num):
    PointP = np.zeros((4,3))
    for B_num in range(Ball_num):
        cosTheta = np.dot(OP_unit[L_num][B_num,:],Vos[B_num,:].T)
        #print(cosTheta)
        
        a = 1
        b = -2*(r/sinBeta[B_num])*cosTheta
        c = -r**2+(r/sinBeta[B_num])**2
        #print(quadratic(a,b,c), np.min(quadratic(a,b,c)))
        OP_module = np.min(quadratic(a,b,c))
        #OP_module = scipy.optimize.fsolve(lambda x: a*x**2 + b*x + c, 0)
        #print(OP_module)
        PointP[B_num,:] = OP_unit[L_num][B_num,:]*OP_module + camera_world
    PointP_real.append(PointP)
print(PointP_real,"\n",PointS_world)
'''
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
for L_num in range(Light_num):
    ax.scatter(PointP_real[L_num][:,0],PointP_real[L_num][:,1],
               PointP_real[L_num][:,2], s=100, c='y')
ax.scatter(PointS_world[:,0],PointS_world[:,1],PointS_world[:,2], s=100, c='r')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
'''
#结果还是有问题
Light_world = np.zeros((Light_num,3)) #光源世界坐标
for L_num in range(Light_num):
    A_matrix = np.zeros((2*Ball_num,3))
    B_matrix = np.zeros(2*Ball_num)
    for B_num in range(Ball_num):
        SP_vec = PointP_real[L_num][B_num,:] - PointS_world[B_num,:]
        SP_unit = SP_vec/norm(SP_vec)
        #Light_vec = (2*(SP_unit*(-OP_unit[L_num][B_num,:]))
         #            *SP_unit+OP_unit[L_num][B_num,:])
         #L = 2(N*R)*N-R，不知道为什么这个方法出来的结果有误，是不是向量方向的问题
        R = cv2.Rodrigues(SP_unit)[0]
        Light_vec = np.dot(R,OP_unit[L_num][B_num,:].T).reshape(3)
        #print(Light_vec)
        Light_unit = -Light_vec/norm(Light_vec)
        #print(Light_unit)

        a1,a2,a3 = Light_unit
        Px, Py, Pz = PointP_real[L_num][B_num,:]
        A1 = np.array([[a2,-a1,0],[a3,0,-a1]])
        B1 = np.array([a2*Px-a1*Py,a3*Px-a1*Pz])
        A_matrix[2*B_num:2*B_num+2,:] = A1
        B_matrix[2*B_num:2*B_num+2] = B1
    Light_world[L_num,:] = pinv(A_matrix).dot(B_matrix)
print(Light_world)
np.save("./light_world.npy",Light_world)

#=======将标定好的空间坐标显示在3D图中
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax.scatter(PointS_world[:,0],PointS_world[:,1],PointS_world[:,2], s=5000, c='r')
ax.scatter(Light_world[:,0],Light_world[:,1],Light_world[:,2], s=100, c='y')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()



    
