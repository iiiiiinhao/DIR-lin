#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:27:14 2018

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable

# from ext import warp
import warp
#from utils import train_utils as tn_utils

class conv_block(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm3d(outChan),
                nn.LeakyReLU(0.2, inplace=True)
#                nn.ReLU(inplace=True)
                )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
#                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                nn.init.xavier_uniform_(m.weight)
                #default: mode='fan_in', nonlinearity='leaky_relu'
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)

        return x

class Unet(nn.Module):
    def __init__(self, enc_nf=[2,16,32,32,32,32], dec_nf=[32,32,32,32,16,16,3]):
        super(Unet, self).__init__()
        """
        unet architecture(voxelmorph-2). 
        :param enc_nf: list of encoder filters. right now it needs to be 1x6. 
            e.g. [2,16,32,32,32,32]
        :param dec_nf: list of decoder filters. right now it must be 1x7
            e.g. [32,32,32,32,16,16,3]
        """
        self.inconv = conv_block(enc_nf[0], enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.up1 = conv_block(enc_nf[-1], dec_nf[0])
        self.up2 = conv_block(dec_nf[0]+enc_nf[4], dec_nf[1])
        self.up3 = conv_block(dec_nf[1]+enc_nf[3], dec_nf[2])
        self.same_conv1 = conv_block(dec_nf[2]+enc_nf[2], dec_nf[3])
        self.up4 = conv_block(dec_nf[3], dec_nf[4])
        self.same_conv2 = conv_block(dec_nf[4]+enc_nf[1], dec_nf[5])
        self.outconv = nn.Conv3d(
                dec_nf[5], dec_nf[6], kernel_size=3, stride=1, padding=1, bias=True)
#        self.tanh = nn.Tanh()
        #init last_conv
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        # down-sample path (encoder)
        skip1 = self.inconv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        # up-sample path (decoder)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip4), 1)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip3), 1)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip2), 1)
        x = self.same_conv1(x)
        x = self.up4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip1), 1)
        x = self.same_conv2(x)
        x = self.outconv(x)

        return x

class dirnet(nn.Module):
    def __init__(self, img_size=[192,256,112], enc_nf=[2,16,32,32,32,32], dec_nf=[32,32,32,32,16,16,3]):
        super(dirnet, self).__init__()
        self.unet = Unet(enc_nf, dec_nf)
        self.warper = warp.Warper3d(img_size)

    def forward(self, mov, ref):
        input0 = torch.cat((mov, ref), 1)
        flow = self.unet(input0)
        warped = self.warper(mov, flow)

        return warped, flow
        
class conv_down(nn.Module):
    """
    Conv3d:三维卷积层, 输入的尺度是(N, C_in,D,H,W)，输出尺度（N,C_out,D_out,H_out,W_out）
    BatchNorm3d:在每一个小批量（mini-batch）数据中，计算输入各个维度的均值和标准差。gamma与beta是可学习的大小为C的参数向量（C为输入大小）
    在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。
    在验证时，训练求得的均值/方差将用于标准化验证数据。
    """

    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(conv_down, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(outChan),
                nn.ReLU(inplace=True)
                )
        self.pool = nn.AvgPool3d(pool_kernel)
#        self.pool = nn.MaxPool3d(pool_kernel)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                nn.init.xavier_uniform_(m.weight)
                #default: mode='fan_in', nonlinearity='leaky_relu'
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv(x)
        if self.down:
            x = self.pool(x)
        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        print(x.shape)
        m_batchsize, C, W, H, D = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, W * H * D).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, W * H * D)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, W * H * D)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, W, H, D)

        out = self.gamma * out + x
        return out, attention


class Net(nn.Module):
    def __init__(self, ndown=3, nfea=[2,16,32,64,64,64,128,64,32,3]):
        super(Net, self).__init__()
        """
        net architecture. 
        :param nfea: list of conv filters. right now it needs to be 1x8.
        :param ndown: num of downsampling, 3 or 4.
        """
        self.ndown = ndown
        assert ndown in [3, 344, 4]
        if ndown == 344:
            self.down1 = conv_down(nfea[0], nfea[1], pool_kernel=(1,2,2))            
        else:
            self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])
        if ndown in [344, 4]:
            self.down4 = conv_down(nfea[3], nfea[3])
            self.same0 = conv_down(nfea[3], nfea[3], down=False)
        self.same1 = conv_down(nfea[3], nfea[4], down=False)
        self.same2 = conv_down(nfea[4], nfea[5], down=False)
        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)
        self.same5 = conv_down(nfea[7], nfea[8], down=False)
        self.outconv = nn.Conv3d(
                nfea[8], nfea[9], kernel_size=1, stride=1, padding=0, bias=True)
        #init last_conv
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        scale=8
        # print('x',x.shape)
        x = F.instance_norm(self.down1(x))
        # print('x1',x.shape)
        x = self.down2(x)
        # print('x2', x.shape)
        x = self.down3(x)
        # print('x3', x.shape)
        if self.ndown in [344, 4]:
            scale=16 if self.ndown==4 else (8,16,16)
            x = self.down4(x)
            x = self.same0(x)
        x = self.same1(x)
        # print('x_s1', x.shape)
        x = self.same2(x)
        # print('x_s2', x.shape)
        x = self.same3(x)
        # print('x_s3', x.shape)
        x = self.same4(x)
        # print('x_s4', x.shape)
        x = self.same5(x)
        # print('x_s5', x.shape)
        x = self.outconv(x)
        # print('x', x.shape)
        # x = F.interpolate(x, scale_factor=[8,8,8.56], mode='trilinear', align_corners=True)  # False
        x = F.interpolate(x, scale_factor=scale, mode='trilinear', align_corners=True)
        # print('x', x.shape)


        return x

class snet(nn.Module):
    def __init__(self, ndown=3, img_size=[256,256,96]):
    # def __init__(self, ndown=3, img_size=[128, 144, 128]):
        super(snet, self).__init__()
        self.net = Net(ndown)
        self.warper = warp.Warper3d(img_size)


    def forward(self, mov, ref):
        input0 = torch.cat((mov, ref), 1)
        input0 = input0.to(torch.float)
        # print(input0.shape)
        flow = self.net(input0)

        warped = self.warper(mov, flow)

        return warped, flow

#a=snet(ndown=344, img_size=[32,32,32])
#in1 = torch.rand((2,1,32,32,32))
#in2 = torch.rand((2,1,32,32,32))
#b,c=a(in1, in2)