# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:23:45 2020

@author: Administrator
"""
import torch
import numpy as np


mean_tot = 0
std_tot = 0
tre_all = {}
for case in range(1,11):
    val = 6
    en = 1
    # spacing_arr = [[0.97,0.97,1.25],[1.16,1.16,1.25],[1.15,1.15,1.25],[1.13,1.13,1.25],[1.10,1.10,1.25],[0.97,0.97,1.25],[0.97,0.97,1.25],
    #                                 [0.97,0.97,1.25],[0.97,0.97,1.25],[0.97,0.97,1.25]]
    spacing_arr = [[0.97, 0.97, 2.5], [1.16, 1.16, 2.5], [1.15, 1.15, 2.5], [1.13, 1.13, 2.5], [1.10, 1.10, 2.5],
                   [0.97, 0.97, 2.5], [0.97, 0.97, 2.5],
                   [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5]]

    spacing = np.array(spacing_arr[case-1])

    # spacing = np.array([2.5,0.97,0.97])
    lmk_path =  'E:/DIR-lin_t/MIR_3D/mark\Case%gPack/ExtremePhases/'%case   # 路径
    mov_lmk_fname = 'Case%g_300_T00_xyz.txt'%case  #实验T00到T50进行配准
    ref_lmk_fname = 'Case%g_300_T50_xyz.txt'%case

    # RESULTS_PATH = 'E:/DIR-lin_t/MIR_3D/results/'
    # flow_folder = RESULTS_PATH + 'flow/'
    flow_folder = 'C:/Users/Administrator/Desktop/loss/loss1.82-lr0.0001-epo100-mse0.2/'
    #%% load flow
    lamb = 10

    #save flow
    if en:
        flow_fname = 'flow_en_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val,case, lamb)
    else:
        flow_fname = 'flow_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val, case, lamb)


    flow = np.load(flow_folder+flow_fname)
    flow = torch.Tensor(flow)


    #flow参数
    H, W, D = 248,160,192

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


    #网格归一化
    xx_f = (xx_f + 1) * H / 2
    yy_f = (yy_f + 1) * W / 2
    zz_f = (zz_f + 1) * D / 2



    mov_lmk = np.loadtxt(lmk_path+mov_lmk_fname)
    ref_lmk = np.loadtxt(lmk_path+ref_lmk_fname)

    # resize_axis = [[242,156,94],[256,176,111],[256,166,104],[256,172,99],[244,173,106],[307,193,128],[299,208,136],[283,220,128],[277,206,128],[238,220,120]]
    # resize_axis = [[242, 155, 89], [256, 176, 109], [256, 165, 102], [256, 176, 97], [244, 170, 104], [306, 194, 119],
    #                [299, 194, 123], [285, 221, 127], [270, 206, 128], [238, 218, 103]]
    resize_axis = [[242,155,89],[256,176,109],[256,165,102],[256,176,97],[244,170,104],[306,194,119],[299,194,123],[285,221,127],[270,206,128],[238,218,103]]
    a = H/resize_axis[case-1][0]
    b = W/resize_axis[case-1][1]
    # c = D/resize_axis[case-1][2]
    # c = 96 / resize_axis[case - 1][2]
    c = D / resize_axis[case - 1][2]
    resize_factor = np.array([a,b,c])

    # print(resize_factor)

    mov_lmk0 = np.zeros([300,3])
    ref_lmk0 = np.zeros([300,3])

    # offset_arr = [[14,42,0],[0,25,0],[0,41,0],[0,36,0],[12,53,0],[125,134,0],[118,139,0],[127,118,0],[111,79,0],[134,118,0],[146,112,0]]
    offset_arr = [[14, 42, 0], [0, 25, 0], [0, 41, 0], [0, 36, 0], [12, 53, 0], [125, 134, 0], [118, 139, 0],
                  [111, 79, 0], [134, 118, 0], [146, 112, 0]]


    # offset_arr = offset_arr[]
    # mov_lmk0 = (mov_lmk - offset_arr[case-1] - 1) * resize_factor*2-1
    # ref_lmk0 = (ref_lmk - offset_arr[case-1] - 1) * resize_factor*2-1
    # mov_lmk0 = (mov_lmk - offset_arr[case-1] - 1) * resize_factor
    # ref_lmk0 = (ref_lmk - offset_arr[case-1] - 1) * resize_factor
    # print(ref_lmk0)
    mov_lmk0[:,0] = (mov_lmk[:,0] - offset_arr[case-1][0] - 1) * resize_factor[0]
    mov_lmk0[:,1] = (mov_lmk[:,1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
    mov_lmk0[:,2] = ((mov_lmk[:,2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
    # mov_lmk0[:, 2] = (mov_lmk[:, 2] - offset_arr[case - 1][2] - 1) * resize_factor[2] * 2 - 1
    ref_lmk0[:,0] = (ref_lmk[:,0] - offset_arr[case-1][0] - 1) * resize_factor[0]
    ref_lmk0[:,1] = (ref_lmk[:,1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
    ref_lmk0[:,2] = ((ref_lmk[:,2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
    # ref_lmk0[:, 2] = (ref_lmk[:, 2]  - offset_arr[case - 1][2] - 1) * resize_factor[2] * 2 - 1
    # print(ref_lmk0)
    # print(resize_factor)


    ref_lmk_index = np.round(ref_lmk0).astype('int32')    #取整
    ref_lmk1 = ref_lmk0.copy()
    ref_lmk_index1 = np.zeros([300,3])
    ref_lmk_index1 = ref_lmk_index      #预留网格取整后被忽略的小数

    for i in range(300):
        hi, wi, di = ref_lmk_index[i]


        h0 = xx_f[hi, wi, di]
        w0 = yy_f[hi, wi, di]
        d0 = zz_f[hi, wi, di]
        ref_lmk1[i] = [h0, w0, d0]



    spacing1 = spacing
    spacing1 = spacing/resize_factor  #转换分辨率



    factor1 = np.ones([300,3])
    factor1 = ref_lmk_index1/ref_lmk1
    ref_lmk1_xs = (ref_lmk0 - ref_lmk_index1)*factor1  #计算网格时被忽略的小数



    # print(ref_lmk1_xs)
    diff1 = (ref_lmk1 - mov_lmk0 + ref_lmk1_xs) * spacing1
    diff1 = torch.Tensor(diff1)
    tre1 = diff1.pow(2).sum(1).sqrt()
    mean1 = tre1.mean()
    std1 = tre1.std()
    mean_tot += mean1
    std_tot += std1
    print('case%g'%case,mean1,'    case%g'%case,std1)



mean_tot = mean_tot/10
std_tot = std_tot/10
print('mean_tot',mean_tot,'    std_tot',std_tot)
