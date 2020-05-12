import os
import  cv2
import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np
import yaml
f = open('gt.yml')
dataMap = yaml.load(f)
image_number=len(dataMap)
image_path="D:\\6Ddataset\\Linemod_preprocessed\\Linemod_preprocessed\\data\\01\\rgb"
image_files=os.listdir(image_path)
file_dir = 'obj_01.ply'  #文件的路径
plydata = PlyData.read(file_dir)  # 读取文件
data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
points = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    points[:, i] = data_pd[name]
#下面要获得8个顶点
#先获得Xmin,Xmax,Ymin,Ymax,Zmin,Zmax
points_size=8
points=np.array(points)
#points_new=points[:,0:3]
points_x=points[:,0]
points_y=points[:,1]
points_z=points[:,2]
Xmin=points_x.min()
Xmax=points_x.max()
Ymin=points_y.min()
Ymax=points_y.max()
Zmin=points_z.min()
Zmax=points_z.max()
points_new=[]
points_new.append([Xmin,Ymin,Zmin])
points_new.append([Xmax,Ymin,Zmin])
points_new.append([Xmax,Ymax,Zmin])
points_new.append([Xmin,Ymax,Zmin])
points_new.append([Xmin,Ymin,Zmax])
points_new.append([Xmax,Ymin,Zmax])
points_new.append([Xmin,Ymax,Zmax])
points_new.append([Xmax,Ymax,Zmax])
points_new=np.array(points_new)
points4_=np.zeros([points_size,4],dtype=float)
points4_[:,0:3]=points_new[:,:]
points4_[:,3]=np.ones(points_size,dtype=float)
points4_=points4_.transpose()
camera_Instrinic = np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0])
camera_Instrinic = camera_Instrinic.reshape([3, 3])
# print("--------")
# print(dataMap[0])
# print("--------")
# print(dataMap[0][0])
for i in range(image_number):
    image=cv2.imread(os.path.join(image_path,image_files[i]))
    Rotation_Matrix=np.array(dataMap[i][0]['cam_R_m2c'])
    Translation_Matrix=np.array(dataMap[i][0]['cam_t_m2c'])
    Rotation_Matrix=Rotation_Matrix.reshape([3,3])
    Translation_Matrix=Translation_Matrix.reshape([3,1])
    KT_Matrix=np.zeros([3,4],dtype=np.float)
    KT_Matrix[0:3,0:3]=Rotation_Matrix[:,:]
    KT_Matrix[0:3,3]=Translation_Matrix[:,0]
    In_KT_multi=np.dot(camera_Instrinic,KT_Matrix)
    three_dim_points=np.dot(In_KT_multi,points4_)
    three_dim_points=np.array(three_dim_points)
    two_dim_points=np.zeros((2,points_size),dtype=float)
    #print(three_dim_points[0:2,:])
    two_dim_points[0,:]=three_dim_points[0,:]/three_dim_points[2,:]
    two_dim_points[1,:]=three_dim_points[1,:]/three_dim_points[2,:]
    #接下来把点画在图片上
    for j in range(8):
        cv2.circle(image, (int(two_dim_points[0,j]),int(two_dim_points[1,j])), 4, (255,255,0))
    cv2.line(image,(int(two_dim_points[0,0]),int(two_dim_points[1,0])),(int(two_dim_points[0,1]),int(two_dim_points[1,1])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,0]),int(two_dim_points[1,0])),(int(two_dim_points[0,3]),int(two_dim_points[1,3])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,0]),int(two_dim_points[1,0])),(int(two_dim_points[0,4]),int(two_dim_points[1,4])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,1]),int(two_dim_points[1,1])),(int(two_dim_points[0,5]),int(two_dim_points[1,5])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,1]),int(two_dim_points[1,1])),(int(two_dim_points[0,2]),int(two_dim_points[1,2])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,2]),int(two_dim_points[1,2])),(int(two_dim_points[0,3]),int(two_dim_points[1,3])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,2]),int(two_dim_points[1,2])),(int(two_dim_points[0,7]),int(two_dim_points[1,7])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,3]),int(two_dim_points[1,3])),(int(two_dim_points[0,6]),int(two_dim_points[1,6])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,4]),int(two_dim_points[1,4])),(int(two_dim_points[0,5]),int(two_dim_points[1,5])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,4]),int(two_dim_points[1,4])),(int(two_dim_points[0,6]),int(two_dim_points[1,6])),color=(255,0,0),thickness=2)
    cv2.line(image,(int(two_dim_points[0,5]),int(two_dim_points[1,5])),(int(two_dim_points[0,7]),int(two_dim_points[1,7])),color=(255,0,0),thickness=2)
    cv2.line(image, (int(two_dim_points[0, 6]), int(two_dim_points[1, 6])),(int(two_dim_points[0, 7]), int(two_dim_points[1, 7])), color=(255, 0, 0), thickness=2)
    cv2.imwrite(image_files[i].rstrip(".png")+"boundingBoxResult.jpg",image)
