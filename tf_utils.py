import numpy as np
import tensorflow as tf
import math

def random_mini_batches_GCN(X, Y, L, mini_batch_size, seed):
    
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))#可直接生成一个m维随机排列的数组
    shuffled_X = X[permutation, :]#取X对应于随机数组的行
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))#取Y对应于随机数组的行并reshape为m*205
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order = "F")#取L对应于随机数组的行并reshape为二维，并拉伸成一维数组后再重新按列分配
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order = "F")#取上述二维矩阵的随机数组对应的列并reshape为二维，拉伸成一维数组后再重新按列分配

    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range(0, num_complete_minibatches):#k从0循环到num_complete_minibatches
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]#取shuffled_X的前mini_batch_size行
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]#取shuffled_Y的前mini_batch_size行
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        #取shuffled_L的前mini_batch_size行和列，
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_L)#一个mini_batch等于XYL的级联
        mini_batches.append(mini_batch)#将循环中的所有minibatch添加到一个list
    mini_batch = (X, Y, L) #原始数据XYL组成一个数组
    mini_batches.append(mini_batch)#将原始数据数组也添加到list中
    
    return mini_batches

def random_mini_batches_GCN1(X, X1, Y, L, mini_batch_size, seed):
    
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))#np.random.permutation一个随机排列函数,就是将输入的数据进行随机排列
    #利用次函数对输入数据X、Y进行随机排序，且要求随机排序后的X Y中的值保持原来的对应关系
    shuffled_X = X[permutation, :]
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order = "F")
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order = "F")

    num_complete_minibatches = math.floor(m / mini_batch_size)#返回小于参数x的最大整数,即对浮点数向下取整
    
    for k in range(0, num_complete_minibatches):       
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_X1, mini_batch_Y, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, X1, Y, L) 
    mini_batches.append(mini_batch)
    #随机生成mini-batches
    return mini_batches
        
def random_mini_batches(X1, X2, Y, mini_batch_size, seed):
    
    m = X1.shape[0]
    m1 = X2.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    permutation1 = list(np.random.permutation(m1))
    shuffled_X2 = X2[permutation1, :]
    
    num_complete_minibatches = math.floor(m1/mini_batch_size)
    
    mini_batch_X1 = shuffled_X1
    mini_batch_Y = shuffled_Y
      
    for k in range(0, num_complete_minibatches):        
        mini_batch_X2 = shuffled_X2[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]        
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_single(X1, Y, mini_batch_size, seed):
    
    m = X1.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    #shuffled_X2 = X2[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
        
    for k in range(0, num_complete_minibatches):
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X1, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
