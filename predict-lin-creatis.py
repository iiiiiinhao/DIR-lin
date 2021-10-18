#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:55:47 2019

@author: user
"""
import numpy as np
import time
from pathlib import Path
#import matplotlib.pyplot as plt

import torch
#import torch.optim as optim
#import torch.utils.data as data
import torchvision.transforms

# import networks
#import dataset
from MIR_3D import model
from MIR_3D import model_aspp
import transform
# from utils import train_utils, visual, test_utils
import utils
import visual
import os
import matplotlib.pylab as plt

import loss
import warp
# from ext import loss, warp
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%%
'''dataset loading'''
torch.backends.cudnn.benchmark = True

img_size = [280, 200, 288]
model = model_aspp.snet(ndown=3, img_size=img_size).cuda()

#batch_size = 1
#win_size = [5,5,5]
val_index = 6
# case = 2

WEIGHTS_PATH = 'E:/DIR-lin_t/MIR_3D/weights-adam/'
# WEIGHTS_PATH = 'C:/Users/Administrator/Desktop/loss/loss1.82-lr0.0001-epo100-mse0.2/'
# WEIGHTS_PATH = 'E:/DIR-lin_t/MIR_3D/result/'
weights_fname = 'weights-val8_plus-100-0.899-0.898.pth'

#root = '../dir/data3/'
for case in range(1,7):
    Transform = torchvision.transforms.Compose([transform.OneNorm(),
                                                transform.ToTensor()])
    path = 'E:/DIR-lin_t/MIR_3D/creatis/val/'
    # if en:
    #     path = 'E:/DIR-lin/MIR_3D/mat_en/case%g/' % case
    # else:
    #     path = 'E:/DIR-lin/MIR_3D/mat/case%g/' % case
    mov_fname = 'case%g_T10.npy' % case
    ref_fname = 'case%g_T60.npy' % case

    mov = np.load(path + mov_fname)
    ref = np.load(path + ref_fname)

    mov = np.expand_dims(mov, 0)#shape(1, D, H, W)
    ref = np.expand_dims(ref, 0)

    mov0 = Transform(mov)
    ref0 = Transform(ref)

    mov = mov0.unsqueeze(0).cuda()
    ref = ref0.unsqueeze(0).cuda()



    #%% weights

    '''trilinear, no seg, reso 1, lcc+mse'''

    #lambda 0, bs 4, win 5, val 8


    startEpoch = utils.load_weights(model, WEIGHTS_PATH + weights_fname)
    #%% '''predict'''
    it = 1
    warper = warp.Warper3d(img_size)
    flow = torch.zeros([1,3]+img_size).cuda()

    #evaluate model for it times
    model.eval()
    since = time.time()
    with torch.no_grad():
        # warped = mov
        # for _ in range(it):
        #     _, flow0 = model(warped, ref)
        #     flow -= flow0
        #     warped = warper(mov, flow)


        _, flow0 = model(mov, ref)
        flow += flow0
        warped = warper(mov, flow)

        # _, flow0 = model(warped, ref)
        # flow -= flow0


    # with torch.no_grad():
    #     warped = mov
    #     for _ in range(it):
    #         _, flow0 = model(warped, ref)
    #         flow += flow0
        # flow += flow0


        # warped = warper(mov, flow)

    time_elapsed = time.time() - since
    print('Prediction Time {:.0f}m {:.04f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    m = mov.data.cpu().numpy()[0,0]
    r = ref.data.cpu().numpy()[0, 0]
    w = warped.data.cpu().numpy()[0, 0]


    # print('m=',m.shape,'r=',r.shape,'w=',w.shape)
    flow = flow.data.cpu()


    visual.view_diff(m, r, w)
    # %% save warped, flow, jac
    # diff1 = abs(w - r)
    diff1 = torch.from_numpy(w)
    # diff0 = abs(m - r)
    diff0 = torch.from_numpy(m)
    # visual.corr_plot(diff0,diff1)
    lamb = 10

    RESULTS_PATH = 'results/creatis_flow/'
    flow_folder = RESULTS_PATH + 'flow/'

    Path(flow_folder).mkdir(exist_ok=True)

    #save flow

    flow_fname = 'flow_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val_index,case, lamb)
    np.save(flow_folder+flow_fname, flow.numpy())
