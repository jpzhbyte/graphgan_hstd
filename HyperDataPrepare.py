import numpy as np
import numpy.linalg
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat

#SAM光谱角
def SAM(x,y):
    a = sum(x*y)
    b = numpy.linalg.norm(x) * numpy.linalg.norm(y)

    return math.acos(  a / b )

#SAM光谱角
def OPD(x,y,bands):
    I = np.eye(bands)
    xt = x.reshape(-1,1)
    xt = xt.transpose()
    yt = y.reshape(-1,1)
    yt = yt.transpose()
    Px_perp = I - x * (1/(xt*x)) * xt
    Py_perp = I - y * (1/(yt*y)) * yt
    OPD = math.sqrt(xt*Py_perp*x + yt*Px_perp*y)
    return OPD

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,d,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    centroids[1,:] = d
    for i in range(k-1):
        index = int(np.random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids

# k均值聚类
def KMeans(dataSet,d,k):
    m = np.shape(dataSet)[0]  #行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True
    # 第1步 初始化centroids
    centroids = randCent(dataSet,d,k)
    while clusterChange:
        clusterChange = False
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100
            minIndex = -1
            # 遍历所有的质心
            #第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = SAM(centroids[j,:],dataSet[i,:])
                #distance = OPD(centroids[j,:],dataSet[i,:],bands)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #第 4 步：更新质心
        for j in range(k-1):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值
    print("Congratulations,cluster complete!")
    return centroids,clusterAssment

#hyp_img = loadmat('./dataset/sandiego_sub.mat')
hyp_img = loadmat('./Data/HYDICE_data.mat')
#hyp_img = loadmat('D://Newfolder//TD//Results//HYDICE//0607_6.mat')
data = hyp_img['data']
target = hyp_img['d']
data_map = hyp_img['map']
plt.matshow(data_map)
plt.show()

d = target.reshape(-1,1)
d = d.transpose()
height,width,bands = data.shape
data = np.reshape(data,[height*width,bands])
k = 2
centroids,clusterAssment = KMeans(data,d,k)
print(data.shape)
print(d.shape)
Kmeans_rst = np.reshape(clusterAssment[:,0],[height,width])
plt.matshow(Kmeans_rst)
plt.show()
#将结果保存为.mat文件
#savemat('./kmeans/sandiego_sub-Kmeans.mat',{'Kmeans_rst': Kmeans_rst})
savemat('./Input/HYDICE/initial_det.mat',{'Kmeans_rst': Kmeans_rst})
print("finished")

# for i in range(99):
#     hyp_img = loadmat('D://Newfolder//TD//Input_data//HYDICE//HYDICE_10.mat')
#     data = hyp_img['data_1']
#     target = hyp_img['d']
#     data_map = hyp_img['map']