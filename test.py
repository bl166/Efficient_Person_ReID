# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse,os

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name', default='MaskResNet50-duke-bs+32-lr+0.1-ds+40', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--rerun', action='store_true', help='rerun feature extraction' )

parser.add_argument('--file_name', default='ckpt-prune-sp+0.9000.pth', type=str, help='save model path')
parser.add_argument('--mat_name', default='pytorch_result.mat', type=str, help='save mat path')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn

import numpy as np
import time
import os
import scipy.io
from glob import glob
from shutil import copyfile

from models.model_legacy import ft_net, ft_net_dense
from utils.evaluate import extract_feature, get_id
from utils.func import *


gpu_ids = [int(g) for g in opt.gpu_ids.split(',') if int(g)>=0]
device = 'cuda' if len(gpu_ids)>0 and torch.cuda.is_available() else 'cpu'

name = opt.model_name
mat_name = opt.mat_name

model_result = os.path.join('./checkpoint',name,mat_name)
model_result_multi = os.path.join('./checkpoint',name,mat_name)


# get model architecture
if 'dense' in name.lower():
    archi_flag = 'den'
elif 'res' in name.lower():
    archi_flag = 'res'
else:
    raise ValueError('Unrecignized network naming')

if not opt.rerun:
    if os.path.isfile(model_result):
        if not os.path.exists('./.tmp'):
            os.mkdir('./.tmp')
#         copyfile(model_result, './.tmp/'+mat_name)
        copyfile(model_result, './.tmp/pytorch_result.mat')
        print('Loading from {}..'.format(model_result))
        quit()
    
# get test data dir
if 'market' in name.lower():
    data_dir = 'Market-1501-v15.09.15/pytorch/'
    nclasses = 751
elif 'duke' in name.lower():
    data_dir = 'DukeMTMC-reID/pytorch/'
    nclasses = 702
else:
    raise ValueError('Cannot recognize the dataset. Market or Duke?')

    
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])


if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes

######################################################
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)
    

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if archi_flag.lower() == 'den':
    model_structure = ft_net_dense(nclasses)
elif archi_flag.lower() == 'res':
    model_structure = ft_net(nclasses)
else:
    raise ValueError

    
file_name = 'ckpt.pth' if opt.file_name is None else opt.file_name
model_path = os.path.join('./checkpoint',name,file_name)
model,_,_ = load_network(model_structure, model_path)
model.to(device)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()

# Change to test mode
if device=='cuda' and len(gpu_ids) > 1:
    model = torch.nn.DataParallel(model)#, device_ids=gpu_ids)
    cudnn.benchmark = True

model.eval()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'], archi_flag, device)
query_feature = extract_feature(model,dataloaders['query'], archi_flag, device)
if opt.multi:
    mquery_feature = extract_feature(model,dataloaders['multi-query'], archi_flag, device)
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,
          'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
# save to specifc model dir
scipy.io.savemat(model_result,result)
# save to project root dir
scipy.io.savemat('./.tmp/pytorch_result.mat',result)
if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat(model_result_multi,result)
    scipy.io.savemat('./.tmp/pytorch_result.mat',result)
