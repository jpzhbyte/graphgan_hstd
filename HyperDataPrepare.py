import numpy as np
import numpy.linalg
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat

def SAM(x,y):
    a = sum(x*y)
    b = numpy.linalg.norm(x) * numpy.linalg.norm(y)

    return math.acos(  a / b )

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

def randCent(dataSet,d,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    centroids[1,:] = d
    for i in range(k-1):
        index = int(np.random.uniform(0,m))
        centroids[i,:] = dataSet[index,:]
    return centroids

def KMeans(dataSet,d,k):
    m = np.shape(dataSet)[0]

    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True

    centroids = randCent(dataSet,d,k)
    while clusterChange:
        clusterChange = False

        for i in range(m):
            minDist = 100
            minIndex = -1

            for j in range(k):

                distance = SAM(centroids[j,:],dataSet[i,:])

                if distance < minDist:
                    minDist = distance
                    minIndex = j

            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2

        for j in range(k-1):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
            centroids[j,:] = np.mean(pointsInCluster,axis=0)

    return centroids,clusterAssment

hyp_img = loadmat('./Data/HYDICE_data.mat')

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

savemat('./Input/HYDICE/initial_det.mat',{'Kmeans_rst': Kmeans_rst})
