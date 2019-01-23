# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0,1,2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir',default='./Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')

parser.add_argument('--archit',default='DenseNet121', type=str, help='Model architecture: DenseNet121 | ResNet47')

parser.add_argument('--batchsize', '-bs', default=32, type=int, help='batchsize')
parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--decay_step', '-ds', default=40, type=int, help='Weight decay step')
parser.add_argument('--epochs', '-ep', default=60, type=int, help='Weight decay step')

parser.add_argument('--resume', default=None, help='restore from the given checkpoint and finetune' )
parser.add_argument('--customized_name','-cn',default='ckpt', type=str, help='customize your model name. useful in pruned finetune')

opt = parser.parse_args()


# ---------------------------
# set gpu config
import torch
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
gpu_ids = [int(g) for g in opt.gpu_ids.split(',') if int(g)>=0]
device = 'cuda' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu'

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
import re
from glob import glob
from shutil import copyfile

from utils.random_erasing import RandomErasing
from models.model_legacy import ft_net, ft_net_dense

from utils.func import *
if is_interactive():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


version =  torch.__version__


# ---------------------------
# define data dir and flag

data_dir = opt.data_dir
if 'duke' in data_dir.lower():
    data_flag = 'duke'
elif 'market' in data_dir.lower():
    data_flag = 'market'
else:
    raise NotImplemented

# ---------------------------
# define model architecture

arch, dep = list(filter(None,re.split(r'([a-zA-Z]+).*?([0-9]+)', opt.archit)))
if 'dense' in arch.lower():
    archi_flag = 'den'
elif 'res' in arch.lower():
    archi_flag = 'res'
else:
    raise NotImplemented
dep = int(dep)

# ---------------------------
# training hyper-params config

bs = opt.batchsize
lr = opt.learning_rate
ds = opt.decay_step

# ---------------------------
# construct model dirname
name = '{}-{}-bs+{}-lr+{}-ds+{}'.format(opt.archit, data_flag, bs, lr, ds)

resume_path = opt.resume
ckpt_fname = opt.customized_name[:-4] if opt.customized_name[-4:]=='.pth' else opt.customized_name
if resume_path is None:
    resume_path = os.path.join('./checkpoint', name, ckpt_fname+'-init.pth')
ckpt_fname = ckpt_fname+'.pth'
    
######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

# print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
# print(time.time()-since)


######################################################################
# Training the model

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, 
                num_epochs=25, start_epoch=0, start_step=0, model_name=name, best_acc=0, logger=None):
    
    def BN_grad_zero():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                mask = (m.weight.data != 0)
                mask = mask.float().to(device)
                m.weight.grad.data.mul_(mask)
                m.bias.grad.data.mul_(mask)
            
    step = start_step
    since = time.time()

#     best_model_wts = model.state_dict()

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in tqdm(dataloaders[phase], disable = phase=='val'):
                                
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
#                 if now_batch_size < bs: # skip the last batch
#                     continue

                # wrap them in Variable
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    BN_grad_zero() # enable this function if not using channel_selection
                    optimizer.step()
                    
                # statistics
                if int(version[2]) > 3: # for the new version like 0.4.0 and 0.5.0
                    loss_ver = loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    loss_ver = loss.data[0] * now_batch_size
                    
                if phase=='train':    
                    step += 1

                running_loss += loss_ver
                running_corrects += float(torch.sum(preds == labels.data))
        
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
                        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                        
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
                        
            if phase == 'val':
                if epoch_acc > best_acc: #epoch%10 == 9:
                    best_acc = epoch_acc
                    save_network(model, epoch, best_acc, './checkpoint/%s/%s'%(model_name,ckpt_fname))
                              
        logger.add_scalars('data/loss', {'train': y_loss['train'][-1],
                                         'val': y_loss['val'][-1]}, epoch)
        logger.add_scalars('data/accuracy', {'train': 1-y_err['train'][-1],
                                             'val': 1-y_err['val'][-1]}, epoch)
        logger.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'],epoch)
        
        for pname, param in model.named_parameters():
            if pname.split('.')[-1].lower()=='weight':
                logger.add_histogram(pname, param.clone().cpu().data.numpy(), epoch)

        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    with open('./checkpoint/%s/.done'%model_name, 'a') as f: 
        f.write('\n'+ckpt_fname)

    return model


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if archi_flag == 'den':
    net = ft_net_dense(len(class_names))
#     net = densenet(depth = dep, nClass = len(class_names))
elif archi_flag == 'res':
    net = ft_net(len(class_names))
else:
    raise NotImplemented

# Load state dict
if resume_path and os.path.exists(resume_path):
    print('Pretrained model loaded from ',resume_path)
    net, _, _ = load_network(net, resume_path)

ep_start, acc_start = 0,0
last_check = './checkpoint/%s/%s'%(name,ckpt_fname)
if os.path.exists(last_check):
    net, ep_start, acc_start = load_network(net, last_check)
    print('Resumed from ',last_check,'with epoch %d and accuracy %.4f'%(ep_start, acc_start))
    ep_start += 1
    
ep_start = max(0,ep_start)

# exit if training is already done
if ep_start>0 and os.path.exists('./checkpoint/%s/.done'%name):
    with open('./checkpoint/%s/.done'%name,'r') as f:
        done_files = f.read().split()
        if ckpt_fname in done_files:
            print('Best val Acc:',acc_start)
            sys.exit()


# tensorboard logger
writer = SummaryWriter(log_dir=os.path.join('./checkpoint',name,'logs'))

net = net.to(device)
criterion = nn.CrossEntropyLoss()
ignored_params = list(map(id, net.model.fc.parameters() )) + list(map(id, net.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer_ft = optim.SGD([
         {'params': base_params, 'lr': .1*lr},
         {'params': net.model.fc.parameters(), 'lr': lr},
         {'params': net.classifier.parameters(), 'lr': lr}
     ], weight_decay=5e-4, momentum=0.9, nesterov=True)


    
# Decay LR by a factor of 0.1 every xxxx epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=ds, gamma=0.1)

paral_flag = False
if device=='cuda' and len(gpu_ids) > 1:
    import torch.backends.cudnn as cudnn
    net = torch.nn.DataParallel(net)#, device_ids=gpu_ids)
    cudnn.benchmark = True
    paral_flag = True

######################################################################
# Train and evaluate
#
dir_name = os.path.join('./checkpoint',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

#record every run
copyfile('./train.py', dir_name+'/train.py')

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

net = train_model(net, criterion, optimizer_ft, exp_lr_scheduler, 
                  start_epoch=ep_start, num_epochs=opt.epochs, best_acc=acc_start, logger=writer)

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(dir_name,"all_scalars.json"))
writer.close()