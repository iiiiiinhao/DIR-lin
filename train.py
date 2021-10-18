#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:51:17 2019

@author: user
"""
'''
python=3.5.4, torch=0.4.1, cv2=3.4.2
'''
import time
import argparse
from pathlib import Path
#import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

# import networks
from MIR_3D import dataset
from MIR_3D import model
from MIR_3D import model_aspp
# import dataset
import transform
# import train_utils
import utils

#from utils import dataset_utils
import loss
import os



'''dataset loading'''
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Training codes')
parser.add_argument('-v', '--val', default=8, type=int,
                    help='the case index of validation')
parser.add_argument('-b', '--batch', default=1, type=int,
                    help='batch size')
parser.add_argument('-l', '--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('-e', '--epoch', default=100, type=int,
                    help='training epochs')
parser.add_argument('-d', '--lamb', default=0.2, type=float,
                    help='lambda, balance the losses.')
#parser.add_argument('-a', '--alpha', default=2, type=float,
#                    help='alpha, balance the flow loss.')
parser.add_argument('-w', '--win', default=[5,5,5], type=int,
                    help='window size, in the LCC loss')
# parser.add_argument('-i', '--image', default=[96, 208, 272], type=int,
#                     help='image size')
parser.add_argument('-i', '--image', default=[280, 200, 288], type=int,
                    help='image size')
parser.add_argument('-p', '--pretrained_model', default=False, type=bool,
                    help='pretrained model')

args = parser.parse_args()
en = 0
att = 0
#distinguish the saved losses
optim_label = 'adam'
loss_label = optim_label + '-val%g-bs%g-lr%.4f-lamb%g-win%g-epoch%g'%(
        args.val, args.batch, args.lr, args.lamb, args.win[0], args.epoch)

##pretrained model
WEIGHTS_MODEL_PATH = 'C:/Users/Administrator/Desktop/loss/loss1.82-lr0.0001-epo100-mse0.2/'
WEIGHTS_NAME = 'weights-val8_plus-att-100-0.835-0.835.pth'


WEIGHTS_PATH = 'weights-adam/ '
Path(WEIGHTS_PATH).mkdir(exist_ok=True)
LOSSES_PATH = 'losses/'
Path(LOSSES_PATH).mkdir(exist_ok=True)
RESULTS_PATH = 'results/creatis_flow/'
Path(RESULTS_PATH).mkdir(exist_ok=True)
'''log file'''
f = open(WEIGHTS_PATH + 'README.txt', 'w')

# root = './mat/'
# root = 'E:/dir_lab_pre/dirlab_32x/'
root = './creatis/'
Transform = transforms.Compose([transform.OneNorm(),
                                transform.ToTensor()])
train_dset = dataset.Volumes(root, args.val, train=True, transform=Transform)
val_dset = dataset.Volumes(root, args.val, train=False, transform=Transform)
train_loader = data.DataLoader(train_dset, args.batch, shuffle=False)
val_loader = data.DataLoader(val_dset, args.batch, shuffle=False)

# print("Train dset: %d" %len(train_dset))
# print("Val dset: %d" %len(val_dset))
print("Train dset: %d" %len(train_loader))
print("Val dset: %d" %len(val_loader))



'''Train'''

# model = model.snet(ndown=3, img_size=args.image).cuda()
model = model_aspp.snet(ndown=3, img_size=args.image).cuda()

if args.pretrained_model:
    model.load_state_dict(torch.load(WEIGHTS_MODEL_PATH + WEIGHTS_NAME),strict=False)

val = args.val

optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.9

#optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.95)
#optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=w_decay)#, momentum=0.95
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Epoch_step_size, gamma=Gamma)

'''lcc + grad'''
criterion = {'lcc': loss.LCC(args.win).cuda(),
             'mse': torch.nn.MSELoss().cuda(),
             'Grad': loss.Grad().cuda(),
             'lambda': args.lamb}

losses = []
for epoch in range(1, args.epoch+1):



    since = time.time()
#    scheduler.step()# adjust lr
    ### Train ###
    trn_loss, trn_lcc, trn_mae = utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
        epoch, trn_loss, trn_lcc, trn_mae))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Epoch {:d}\nTrain - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
        epoch, trn_loss, trn_lcc, trn_mae), file=f)      
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)

    ### Val ###
    val_loss, val_lcc, val_mae = utils.test(
            model, val_loader, criterion, epoch)    
    print('Val - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
            val_loss, val_lcc, val_mae))
    time_elapsed = time.time() - since  
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Val - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
            val_loss, val_lcc, val_mae), file=f)
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)

    ### Checkpoint ###    
    utils.save_weights(en, att, model, val, epoch, val_loss, val_lcc, WEIGHTS_PATH)
        
    ### Save/Plot loss ###
    loss_info = [epoch, trn_loss, trn_lcc, val_loss, val_lcc]
    losses.append(loss_info)
    
utils.save_loss(losses, loss_label, LOSSES_PATH, RESULTS_PATH)
f.close()
