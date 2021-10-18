# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:30:12 2020

@author: 起明
"""
import SimpleITK as sitk
import csv
import scipy.io
from PIL import Image

# def loadDCM(filename):
#
#     ds = sitk.ReadImage(filename)
#
#     img_array = sitk.GetArrayFromImage(ds)
#
#     _, width, height = img_array.shape #frame_num
#
#     return img_array[0], width, height


def loadmat(filename):
    data = scipy.io.loadmat(filename)  # 读取mat文件
    img_array = data['en_raw00']
    width, height, _ = img_array.shape  # frame_num
    return img_array[0], width, height



def readCSV(filename):
    lines = []
    with open(filename, "rt", encoding="utf-8") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines