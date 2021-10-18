# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:23:45 2020

@author: Administrator
"""
import torch

import numpy as np
from matplotlib import pyplot as plt
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mean_tot = 0
std_tot = 0
tre_all = {}
for case in range(1,7):
    val = 6
    en = 1
    spacing_arr = [[0.976562, 0.976562, 2.0], [0.976562, 0.976562, 2.0], [0.878906, 0.878906, 2.0],
                   [0.78125, 0.78125, 2.0], [1.17188, 1.17188, 2.0], [1.17188, 1.17188, 2.0]]


    spacing = np.array(spacing_arr[case-1])


    lmk_path =  'E:/creatis_data/case%g/'%case   # 路径
    if case==1:
        mov_lmk_fname = 'case%g_100_T10.txt'%case  #实验T00到T50进行配准
        ref_lmk_fname = 'case%g_100_T60.txt'%case
    else:
        mov_lmk_fname = 'case%g_100_T00.txt' % case  # 实验T00到T50进行配准
        ref_lmk_fname = 'case%g_100_T50.txt' % case

    RESULTS_PATH = 'E:/DIR-lin_t/MIR_3D/results/'
    flow_folder = RESULTS_PATH + 'creatis_flow/flow/'

    #%% load flow
    lamb = 10

    #save flow

    flow_fname = 'flow_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val, case, lamb)


    flow = np.load(flow_folder+flow_fname)
    flow = torch.Tensor(flow)

    #flow参数

    H, W, D = 280,200,288

    xx = torch.arange(0, H).view(-1, 1, 1).repeat(1, W, D)
    yy = torch.arange(0, W).view(1, -1, 1).repeat(H, 1, D)
    zz = torch.arange(0, D).view(1, 1, -1).repeat(H, W, 1)
    xx = 2.0*xx/H -1
    yy = 2.0*yy/W -1
    zz = 2.0*zz/D -1

    xx_f = torch.zeros([H,W,D])
    yy_f = torch.zeros([H,W,D])
    zz_f = torch.zeros([H,W,D])

    xx_f[:,:,:] = flow[0,2,:,:,:]
    yy_f[:,:,:] = flow[0,1,:,:,:]
    zz_f[:,:,:] = flow[0,0,:,:,:]

    xx_f = xx + xx_f
    yy_f = yy + yy_f
    zz_f = zz + zz_f



    xx_f = (xx_f + 1) * H / 2
    yy_f = (yy_f + 1) * W / 2
    zz_f = (zz_f + 1) * D / 2




    mov_lmk = np.loadtxt(lmk_path+mov_lmk_fname)
    ref_lmk = np.loadtxt(lmk_path+ref_lmk_fname)
    landnum = mov_lmk.shape[0]//3


    resize_axis = [[318,243,141],[316,233,156],[311,204,159],[330,212,187],[179,137,139],[211,174,136]]
    a = H/resize_axis[case-1][0]
    b = W/resize_axis[case-1][1]
    c = D / resize_axis[case - 1][2]
    resize_factor = np.array([a,b,c])

    # print(resize_factor)
    pre_offset_arr = [[-250, -250, -164.5], [-250, -250, -640], [-225, -225, -612.5], [-200, -200, -638],
                      [-300, -300, -546], [-300, -300, -615.5]]
    offset_arr = [[93, 98, 0], [84, 87, 13], [98, 135, 0], [89, 109, 0], [161, 179, 0], [148, 158, 25]]

    mov_lmk0 = np.zeros([landnum,3])
    ref_lmk0 = np.zeros([landnum,3])
    for i in range(0, landnum):
        ref_lmk0[i][0] = ref_lmk[i * 3 + 0]
        ref_lmk0[i][1] = ref_lmk[i * 3 + 1]
        ref_lmk0[i][2] = ref_lmk[i * 3 + 2]
        mov_lmk0[i][0] = mov_lmk[i * 3 + 0]
        mov_lmk0[i][1] = mov_lmk[i * 3 + 1]
        mov_lmk0[i][2] = mov_lmk[i * 3 + 2]

    mov_lmk0[:, 0] = ((mov_lmk0[:, 0] - pre_offset_arr[case - 1][0] - 1) / (spacing_arr[case-1][0]) - offset_arr[case - 1][0]) * resize_factor[0]
    mov_lmk0[:, 1] = ((mov_lmk0[:, 1] - pre_offset_arr[case - 1][1] - 1) / (spacing_arr[case-1][1]) - offset_arr[case - 1][1]) * resize_factor[1]
    mov_lmk0[:, 2] = ((mov_lmk0[:, 2] - pre_offset_arr[case - 1][2] - 1) / (spacing_arr[case-1][2]) - offset_arr[case - 1][2]) * resize_factor[2]
    ref_lmk0[:, 0] = ((ref_lmk0[:, 0] - pre_offset_arr[case - 1][0] - 1) / (spacing_arr[case-1][0]) - offset_arr[case - 1][0]) * resize_factor[0]
    ref_lmk0[:, 1] = ((ref_lmk0[:, 1] - pre_offset_arr[case - 1][1] - 1) / (spacing_arr[case-1][1]) - offset_arr[case - 1][1]) * resize_factor[1]
    ref_lmk0[:, 2] = ((ref_lmk0[:, 2] - pre_offset_arr[case - 1][2] - 1) / (spacing_arr[case-1][2]) - offset_arr[case - 1][2]) * resize_factor[2]

    # mov_lmk0[:, 0] = ((mov_lmk0[:, 0] - pre_offset_arr[case - 1][0] - 1) / (spacing_arr[case - 1][0]) -
    #                   offset_arr[case - 1][0])
    # mov_lmk0_int = mov_lmk0.round()
    # print(mov_lmk0.dtype,mov_lmk0_int.dtype)
    # print(mov_lmk0,mov_lmk0_int)
    # mov_lmk0[:, 1] = ((mov_lmk0[:, 1] - pre_offset_arr[case - 1][1] - 1) / (spacing_arr[case - 1][1]) -
    #                   offset_arr[case - 1][1]) * resize_factor[1]
    # mov_lmk0[:, 2] = ((mov_lmk0[:, 2] - pre_offset_arr[case - 1][2] - 1) / (spacing_arr[case - 1][2]) -
    #                   offset_arr[case - 1][2]) * resize_factor[2]
    # ref_lmk0[:, 0] = ((ref_lmk0[:, 0] - pre_offset_arr[case - 1][0] - 1) / (spacing_arr[case - 1][0]) -
    #                   offset_arr[case - 1][0]) * resize_factor[0]
    # ref_lmk0[:, 1] = ((ref_lmk0[:, 1] - pre_offset_arr[case - 1][1] - 1) / (spacing_arr[case - 1][1]) -
    #                   offset_arr[case - 1][1]) * resize_factor[1]
    # ref_lmk0[:, 2] = ((ref_lmk0[:, 2] - pre_offset_arr[case - 1][2] - 1) / (spacing_arr[case - 1][2]) -
    #                   offset_arr[case - 1][2]) * resize_factor[2]





    ref_lmk_index = np.round(ref_lmk0).astype('int32')
    ref_lmk1 = ref_lmk0.copy()
    ref_lmk_index1 = np.zeros([landnum,3])
    ref_lmk_index1 = ref_lmk_index


    for i in range(landnum):
        hi, wi, di = ref_lmk_index[i]
        # print(i)
        # print(xx_f.shape)
        # print(ref_lmk_index.shape)
        # print(hi,wi,di)

        h0 = xx_f[hi, wi, di]
        w0 = yy_f[hi, wi, di]
        d0 = zz_f[hi, wi, di]
        ref_lmk1[i] = [h0, w0, d0]


    spacing1 = spacing
    spacing1 = spacing/resize_factor

    factor1 = np.ones([landnum,3])

    factor1 = ref_lmk_index1/ref_lmk1  #3
    ref_lmk1_xs = (ref_lmk0 - ref_lmk_index1)*factor1


    # print(ref_lmk1_xs)
    diff1 = (ref_lmk1 - mov_lmk0 + ref_lmk1_xs) * spacing1
    # diff1 = (ref_lmk1 - mov_lmk0 ) * spacing1
    diff1 = torch.Tensor(diff1)
    tre1 = diff1.pow(2).sum(1).sqrt()
    mean1 = tre1.mean()
    std1 = tre1.std()
    mean_tot += mean1
    std_tot += std1
    print('case%g'%case,mean1,'    case%g'%case,std1)
    # tre1 = tre1.numpy()
    # tre_all['case%g'%case] = tre1

# scipy.io.savemat('E:/boxplot/data.mat', tre_all)
# boxplot.Boxplot(tre_all)
mean_tot = mean_tot/6
std_tot = std_tot/6
print('mean_tot',mean_tot,'    std_tot',std_tot)

