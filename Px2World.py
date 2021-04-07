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


#======像素坐标与世界坐标转化====
def pxtoworld(cameraParams,*CoordInPx):
    if len(CoordInPx) == 1:    
        u = CoordInPx[0][:,0]
        v = CoordInPx[0][:,1]        
    elif len(CoordInPx) == 2:
        u = CoordInPx[0]
        v = CoordInPx[1]
    #print(u,v)
    m = cameraParams["M"]
    r = cameraParams["R"]
    t = cameraParams["T"]
    rt = np.hstack((r[:,0:2],t))
    H = m * rt
    #print(H)
    H_11 = H[0,0]
    H_12 = H[0,1]
    H_13 = H[0,2]
    H_21 = H[1,0]
    H_22 = H[1,1]
    H_23 = H[1,2]
    H_31 = H[2,0]
    H_32 = H[2,1]
    H_33 = H[2,2]
    k = len(CoordInPx[0])
    '''
利用矩阵思想加快进程：
[xw]    [u*h31-h11   u*h32-h12]-1       [h13-u*h33]
     =                             *    
[yw]    [v*h31-h21   v*h32-h22]         [h23-v*h33]
扩展开：

    '''
    U0 = u*H_31-H_11
    U1 = u*H_32-H_12
    V0 = v*H_31-H_21
    V1 = v*H_32-H_22
    U2 = H_13-u*H_33
    V2 = H_23-v*H_33
    A = np.hstack((np.vstack((U0,V0)),np.vstack((U1,V1))))
    B = np.vstack((U2,V2))
    #print(A.shape,B.shape)
    X = pinv(A).dot(B)

    X0 = k*np.diag(X[0:k,:])
    Y0 = k*np.diag(X[k:(2*k),:])
    CoordInWorld = np.column_stack((X0,Y0))
    return CoordInWorld
#======进行验证======
def test(cameraParams,*CoordInPx):
    if len(CoordInPx) == 1:    
        u = CoordInPx[0][:,0]
        v = CoordInPx[0][:,1]        
    elif len(CoordInPx) == 2:
        u = CoordInPx[0]
        v = CoordInPx[1]
    m = cameraParams["M"]
    r = cameraParams["R"]
    t = cameraParams["T"]
    rt = np.hstack((r[:,0:2],t))
    H = m * rt
    #print(H)
    H_11 = H[0,0]
    H_12 = H[0,1]
    H_13 = H[0,2]
    H_21 = H[1,0]
    H_22 = H[1,1]
    H_23 = H[1,2]
    H_31 = H[2,0]
    H_32 = H[2,1]
    H_33 = H[2,2]
    k = len(CoordInPx[0])
    
    CoordInWorld = np.zeros((k,2))
    for i in range(k):
        H1 = [[u[i]*H_31-H_11,u[i]*H_32-H_12],[v[i]*H_31-H_21,v[i]*H_32-H_22]]
        H2 = [[H_13-u[i]*H_33],[H_23-v[i]*H_33]]
        CoordInWorld[i,:] = (pinv(H1).dot(H2)).reshape(2)
    return CoordInWorld

def main():
    cameraParams = np.load("C:/Users/Administrator/Desktop/Project/PS/lvbiaoding/cameraParams.npz")
    CoordInPx = np.random.randint(0,10,size=[7,2])
    print(CoordInPx)
    x = pxtoworld(cameraParams,CoordInPx)
    print("x1=\n",x)
    x = test(cameraParams,CoordInPx)
    print("x2=\n",x)
    CoordInPx = np.array([[7,100],[7,1000]])
    x = pxtoworld(cameraParams,CoordInPx)
    print("x3=\n",x)
    CoordInPx = 10*np.ones((2048,2))
    for i in range(2048):
        CoordInPx[i,0] = i
    x = pxtoworld(cameraParams,CoordInPx)
    print("x4=\n",x)        

if __name__=="__main__":
    main()

