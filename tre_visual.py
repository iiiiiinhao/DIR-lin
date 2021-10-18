# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:23:45 2020

@author: Administrator
"""
import torch
from torch.nn import functional as F
import scipy.io as scio
import numpy as np
from matplotlib import pyplot as plt
import os

from boxplot import Boxplot

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def corr_plot(diff_br, diff_ar, mode='tre', title=None, xlabel=None, ylabel=None):
    '''plot x or y correlation scatter
    params:
        tre_br: TRE before registration, ndarray of shape(300, 3)
        tre_ar: TRE after registration, ndarray of shape(300, 3)
        title: the title of plot, default:'Target-Prediction Correlation'
        mode: 'tre' or 'xyz', default:'tre'
    '''
    assert mode in {'tre', 'xyz'}
    if title == None:
        title = 'TRE scatterplot'
    if xlabel == None:
        xlabel = 'TRE before registration (mm)'
    if ylabel == None:
        ylabel = 'TRE after registration (mm)'
    if mode == 'tre':
        tre_br = diff_br.pow(2).sum(1).sqrt()
        tre_ar = diff_ar.pow(2).sum(1).sqrt()
        plt.figure()
        plt.scatter(tre_br, tre_ar, s=5, alpha=0.2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((-2, 33))
        plt.ylim((-2, 33))
        plt.title(title)
    #        plt.grid(True)
    else:
        fig, ax = plt.subplots(1, 3, figsize=[15, 5])
        #        diag_x, diag_y = [-6,6], [-6,6]#diagonal line
        #        ax1.plot(diag_x, diag_y, 'b')
        #        ax2.plot(diag_x, diag_y, 'b')
        ax[0].scatter(diff_br[:, 0], diff_ar[:, 0], s=5, alpha=0.2)
        ax[0].set_title('X direction')
        #        ax[0].set_xlim((-5,5))
        #        ax[0].set_ylim((-5,5))
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        ax[1].scatter(diff_br[:, 1], diff_ar[:, 1], s=5, alpha=0.5)
        ax[1].set_title('Y direction')
        #        ax[1].set_xlim((-5,5))
        #        ax[1].set_ylim((-5,5))
        ax[1].set_xlabel(xlabel)
        #        ax[1].set_ylabel('TRE after registration (mm)')
        ax[2].scatter(diff_br[:, 2], diff_ar[:, 2], s=5, alpha=0.5)
        ax[2].set_title('Z direction')
        #        ax[2].set_xlim((-12,5))
        #        ax[2].set_ylim((-12,5))
        #        fig.suptitle(title)
        ax[2].set_xlabel(xlabel)
    #        ax[1].set_ylabel('TRE after registration (mm)')
    #        ax[0].grid(True)
    #        ax[1].grid(True)
    #        ax[2].grid(True)

    #    plt.tight_layout()
    plt.show()


class tre_visual():
    mean_tot = 0
    std_tot = 0

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
        lmk_path =  'E:/Deformable-Image-Registration-Projects-master/TG-132/Case%gPack/ExtremePhases/'%case   # 路径
        mov_lmk_fname = 'Case%g_300_T00_xyz.txt'%case  #实验T00到T50进行配准
        ref_lmk_fname = 'Case%g_300_T50_xyz.txt'%case

        RESULTS_PATH = 'E:/DIR-lin_t/MIR_3D/results/'

        flow_folder = RESULTS_PATH + 'flow/'
        # flow_folder = 'C:/Users/Administrator/Desktop/loss/loss1.82-lr0.0001-epo100-mse0.2/'
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



        xx_f = (xx_f + 1) * H / 2
        yy_f = (yy_f + 1) * W / 2
        zz_f = (zz_f + 1) * D / 2




        mov_lmk = np.loadtxt(lmk_path+mov_lmk_fname)
        ref_lmk = np.loadtxt(lmk_path+ref_lmk_fname)


        resize_axis = [[242,155,89],[256,176,109],[256,165,102],[256,176,97],[244,170,104],[306,194,119],[299,194,123],[285,221,127],[270,206,128],[238,218,103]]
        a = H/resize_axis[case-1][0]
        b = W/resize_axis[case-1][1]

        c = D / resize_axis[case - 1][2]
        resize_factor = np.array([a,b,c])



        mov_lmk0 = np.zeros([300,3])
        ref_lmk0 = np.zeros([300,3])

        offset_arr = [[14, 42, 0], [0, 25, 0], [0, 41, 0], [0, 36, 0], [12, 53, 0], [125, 134, 0], [118, 139, 0],
                      [111, 79, 0], [134, 118, 0], [146, 112, 0]]



        # print(ref_lmk0)
        mov_lmk0[:,0] = (mov_lmk[:,0] - offset_arr[case-1][0] - 1) * resize_factor[0]
        mov_lmk0[:,1] = (mov_lmk[:,1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
        mov_lmk0[:,2] = ((mov_lmk[:,2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
        ref_lmk0[:,0] = (ref_lmk[:,0] - offset_arr[case-1][0] - 1) * resize_factor[0]
        ref_lmk0[:,1] = (ref_lmk[:,1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
        ref_lmk0[:,2] = ((ref_lmk[:,2] - offset_arr[case - 1][2] - 1) * resize_factor[2])


        ref_lmk_index = np.round(ref_lmk0).astype('int32')
        ref_lmk1 = ref_lmk0.copy()
        ref_lmk_index1 = np.zeros([300,3])
        ref_lmk_index1 = ref_lmk_index

        for i in range(300):
            hi, wi, di = ref_lmk_index[i]

            h0 = xx_f[hi, wi, di]
            w0 = yy_f[hi, wi, di]
            d0 = zz_f[hi, wi, di]
            ref_lmk1[i] = [h0, w0, d0]

        spacing1 = spacing
        spacing1 = spacing/resize_factor



        factor1 = np.ones([300,3])
        factor1 = ref_lmk_index1/ref_lmk1  #3
        ref_lmk1_xs = (ref_lmk0 - ref_lmk_index1)*factor1


        diff1 = (ref_lmk1 - mov_lmk0 + ref_lmk1_xs) * spacing1
        diff1 = torch.Tensor(diff1)
        # print(diff1.shape)
        tre1 = diff1.pow(2).sum(1).sqrt()
        tre = tre1.numpy()
        mean1 = tre1.mean()
        std1 = tre1.std()
        mean_tot += mean1
        std_tot += std1
        print('case%g'%case,mean1,'    case%g'%case,std1)
        # corr_plot(3.68,tre)

        Boxplot(diff1,case)


    mean_tot = mean_tot/10
    std_tot = std_tot/10
    # Boxplot(diff_all)
    # print('mean_tot',mean_tot,'    std_tot',std_tot)



