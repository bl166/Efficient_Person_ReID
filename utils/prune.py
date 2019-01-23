import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import os
from glob import glob
from copy import deepcopy
import numpy as np

import sys; sys.path.append("..")
from models.model_legacy import ft_net, ft_net_dense


def _preproc(model, sparsity, verb = True):
    
    total = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

#     if sparsity<=0:
#         print('Channels: #total = %d, #pruned = 0, pruned ratio = 0, threshold = 0'%total)
#         return model
        
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * sparsity)
    thre = y[thre_index]
    
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            
            if verb:
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, mask.shape[0], int(torch.sum(mask))))
                
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    
    model.cfg = cfg
    model.cfg_mask = cfg_mask
    
    print('Cfg:',cfg)
    if verb:
        print('Cfg:',cfg)
        print('Channels: #total = %d, #pruned = %d, pruned ratio = %.2f, threshold = %.4f'\
              %(total, pruned.numpy(),(pruned/total).numpy(), thre.numpy()))

    return model



def _execute_dense(model, **argw):
    
    newmodel = densenet(**argw)
    cfg_mask = model.cfg_mask
    
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    
    first_conv = True 

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batch normalization layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the mask parameter `indexes` for the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(model.cfg_mask):
                    end_mask = model.cfg_mask[layer_id_in_cfg]
                    
                continue
             

        elif isinstance(m0, nn.Conv2d):
                
            if first_conv:
                # We don't change the first convolution layer.
                m1.weight.data = m0.weight.data.clone()
                first_conv = False 
                continue
                
            if isinstance(old_modules[layer_id - 1], channel_selection):
                                
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#                 print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                # If the last layer is channel selection layer, then we don't change the number of output channels of 
                # the current convolutional layer.
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue         

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            
    return newmodel



def _execute_res(model, **argw):

    newmodel = resnet(**argw)
    cfg_mask = model.cfg_mask

    if 'module' in dir(model):
        model = model.module

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0


    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):

            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):

            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#                 print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the 
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions. 
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            
    return newmodel



    
def prune(model, sparsity = 0, verb = True):
    
    interm_model = _preproc(model.cpu(),sparsity,verb)
        
    if 'dense' in interm_model.name:
        _excute = _execute_dense
    elif 'res' in interm_model.name:
        _excute = _execute_res
    else:
        raise NotImplemented
        
    final_model = _excute(interm_model.cpu(),**interm_model.argws,cfg=interm_model.cfg)
   
    params = [np.sum([param.nelement() for param in m.parameters()]) for m in [model, interm_model, final_model]]
    
    print('#parameters reduced: {:,} -> {:,} -> {:,} ({:%})'.format(*params, (params[0]-params[-1])/params[0]))

    #     torch.save(model.state_dict(), './pruned_%.4f.pth'%sparsity)
    
    return final_model




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('--name',default='ft_DenseNet121', type=str,help='model name')
    parser.add_argument('--base_model',default='', type=str,help='base model name')
    opt = parser.parse_args()
    
    if opt.base_model == '':
        naming = './model/'+opt.name+'/net_{}.pth'
        models = glob(naming.format('*'))
        if not models:
            raise ValueError('Empty model folder!')
        
        ep = [os.path.basename(m).split('.pth')[0].split('net_')[-1] for m in models]
        base_path = naming.format(max(ep))
    else:
        base_path = './model/'+opt.name+'/'+opt.base_model
        
    print('Loading from', base_path, '...')

#     net = ft_net(751)
    net = ft_net_dense(751)
    net.load_state_dict(torch.load(base_path))
    
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    output = net(input)
    print('net output size:', output.shape)
    
    # ------------------
    # prune with sparsity 
    
    sparsity = .4
    pruned_net = prune(net, sparsity)
    
    output = pruned_net(input)
    print('net output size:', output.shape)

    torch.save(model.state_dict(), './model/'+opt.name+'/pruned_%.4f.pth'%sparsity)

