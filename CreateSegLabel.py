import os
import  cv2
import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np

file_dir = 'obj_01.ply'  #文件的路径
plydata = PlyData.read(file_dir)  # 读取文件
data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
points = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    points[:, i] = data_pd[name]
camera_Instrinic=np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0])
camera_Instrinic=camera_Instrinic.reshape([3,3])
Rotation_Matrix=np.array([0.09630630, 0.99404401, 0.05100790, 0.57332098, -0.01350810, -0.81922001, -0.81365103, 0.10814000, -0.57120699])
Translation_Matrix=np.array([-105.35775150, -117.52119142, 1014.87701320])
Rotation_Matrix=Rotation_Matrix.reshape([3,3])
Translation_Matrix=Translation_Matrix.reshape([3,1])
image=cv2.imread("0000.png")
KT_Matrix=np.zeros([3,4],dtype=np.float)
KT_Matrix[0:3,0:3]=Rotation_Matrix[:,:]
KT_Matrix[0:3,3]=Translation_Matrix[:,0]
In_KT_multi=np.dot(camera_Instrinic,KT_Matrix)
points_size=points.shape[0]
points=np.array(points)
points_new=points[:,0:3]
points4_=np.zeros([points_size,4],dtype=float)
points4_[:,0:3]=points_new[:,:]
points4_[:,3]=np.ones(points_size,dtype=float)
points4_=points4_.transpose()
three_dim_points=np.dot(In_KT_multi,points4_)
three_dim_points=np.array(three_dim_points)
two_dim_points=np.zeros((2,points_size),dtype=float)
print(three_dim_points[0:2,:])
two_dim_points[0,:]=three_dim_points[0,:]/three_dim_points[2,:]
two_dim_points[1,:]=three_dim_points[1,:]/three_dim_points[2,:]
#接下来把点画在图片上
for i in range(points_size):
    cv2.circle(image, (int(two_dim_points[0,i]),int(two_dim_points[1,i])), 1, (255,0,0))
cv2.imwrite("result.jpg",image)
k=1