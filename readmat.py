import scipy.io
import matplotlib.pylab as plt
import numpy as np


for a in range(1,7):
    for i in range(0,10):
        data = scipy.io.loadmat('E:/creatis_data/case%d/case%d_raw%d0.mat'%(a,a,i)) # 读取mat文件
        print(data.keys())
        A = data['raw%d0'%i]
        print(A.shape)
        # offset = scipy.io.loadmat('E:/vessel_filter/dir_lab_data/case%d/case%d_offset%d0.mat'%(a,a,i))
        # print(offset.keys())
        # print(offset["sizex1"],offset["sizey1"], offset["sizez1"],
        #       offset["x_l"], offset["y_l"], offset["z_l"])

        # B = np.resize(A,(256,256,96))
        # print(B.shape)
        np.save('E:/dir_lab_pre/creatis/case%d/case%d_T%d0.npy' % (a,a,i), A)
# i = np.save('E:/Deformable-Image-Registration-Projects-master/MIR_3D/mat/case5/case5_T10.npy')
# print(i.shape)
# print(type(i))




