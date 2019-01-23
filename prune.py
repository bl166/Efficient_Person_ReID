
import argparse
import torch

######################################################################
# Options
# --------
# need input of:     model_dir, sparsity, gpu

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='2,3', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='MaskDenseNet121-market-bs+32-lr+0.1-ds+40',type=str, help='model name')
parser.add_argument('--sparsity',default=None, type=float, help='prune sparsity')
parser.add_argument('--check_mode', action='store_true', help='test and eval only' )

# 'MaskDenseNet121-duke-bs+32-lr+0.1-ds+40'
# 'MaskDenseNet121-market-bs+32-lr+0.1-ds+40'
# 'MaskResNet50-duke-bs+32-lr+0.1-ds+40'
# 'MaskResNet50-market-bs+32-lr+0.1-ds+40'

opt = parser.parse_args()

import os
import torch 

os.environ["CUDA_VISIBALE_DEVICE"]=opt.gpu_ids
gpu_ids = [int(g) for g in os.environ["CUDA_VISIBALE_DEVICE"].split(',')]


from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import os
import scipy.io

from utils.evaluate import evaluate, get_id, extract_feature
from utils.func import *
from models.model_legacy import ft_net,ft_net_dense
from utils.prune import _preproc as prune

if is_interactive():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    
    
    
paral_flag = False
device = 'cuda' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu'
if device=='cuda' and len(gpu_ids) > 1:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    paral_flag = True
    

def test_(model, result_dir, data_dir, sparsity=0, rerun=False):
    if os.path.isfile(os.path.join(result_dir,'pytorch_result-sp+%.4f.mat')%sparsity) and not rerun:
        return
    
    data_transforms = transforms.Compose([
            transforms.Resize((288,144), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32,
                                                 shuffle=False, num_workers=16) for x in ['gallery','query']}

    class_names = image_datasets['query'].classes

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)

    if 'module' in dir(model):
        model.module.model.fc = nn.Sequential()
        model.module.classifier = nn.Sequential()
    else:
        model.model.fc = nn.Sequential()
        model.classifier = nn.Sequential()
        
    model = model.eval()#.cuda()    

    # Extract feature
    m_archi = model.module.name[:3] if 'module' in dir(model) else model.name[:3] # dense | res      
    gallery_feature = extract_feature(model,dataloaders['gallery'], m_archi)
    query_feature = extract_feature(model,dataloaders['query'], m_archi)

    result = {'gallery_f':gallery_feature.cpu().numpy(),
              'gallery_label':gallery_label,
              'gallery_cam':gallery_cam,
              'query_f':query_feature.cpu().numpy(),
              'query_label':query_label,
              'query_cam':query_cam}
    
    scipy.io.savemat(os.path.join(result_dir,'pytorch_result-sp+%.4f.mat')%sparsity,result)
    if sparsity <= 0:
        scipy.io.savemat(os.path.join(result_dir,'pytorch_result.mat'),result)
    
    
def eval_(result_path='pytorch_result.mat'):
    result = scipy.io.loadmat(result_path)
    
    query_feature = torch.FloatTensor(result['query_f']).cuda()
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    
    gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

#     print(query_feature.shape)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()/len(query_label) #average CMC
    res_string = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label))
    print(res_string)
    return {'acc':CMC, 'mAP':ap/len(query_label)}


model_name = opt.model_name
model_dir = './checkpoint/'+model_name

if 'market' in model_name.lower():
    data_dir = 'Market-1501-v15.09.15/pytorch/'
    nclass = 751
elif 'duke' in model_name.lower():
    data_dir = 'Market-1501-v15.09.15/pytorch/'
    nclass = 702    
else:
    raise ValueError
    
# ------------------
# prune with sparsity 

sparsity = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5] if opt.sparsity is None else [opt.sparsity]

res_r = []

if not opt.check_mode:
    for s in sparsity:
        print('* PRUNING - Sparsity =',s,'*')

        if "dense" in model_dir.lower():
            model_raw = ft_net_dense(nclass)
        elif "res" in model_dir.lower():
            model_raw = ft_net(nclass)
        else:
            raise ValueError

        model_path = model_dir+'/ckpt.pth'
        model_raw,_,_ = load_network(model_raw, model_path)
#         model_raw.model.fc = nn.Sequential()
#         model_raw.classifier = nn.Sequential()

        model_pru = prune(model_raw, s, verb=False)

        if paral_flag:
            model_pru = torch.nn.DataParallel(model_pru)#, device_ids=gpu_ids)
        model_pru = model_pru.to(device)

        test_(model_pru, model_dir, data_dir, s)
        res = eval_(model_dir+'/pytorch_result-sp+%.4f.mat'%s)

        save_network(model_pru, -1, res['acc'][0], model_dir+'/ckpt-prune-sp+%.4f-init.pth'%s)    
        res_r.append(res)

else:    
    for s in sparsity:
        print('* CHECKING - Sparsity =',s,'*')

        if "dense" in model_dir.lower():
            model_raw = ft_net_dense(nclass)
        elif "res" in model_dir.lower():
            model_raw = ft_net(nclass)
        else:
            raise ValueError

        model_path = model_dir+'/ckpt-prune-sp+%.4f-init.pth'%s
        model_raw,_,_ = load_network(model_raw, model_path)
#         model_raw.model.fc = nn.Sequential()
#         model_raw.classifier = nn.Sequential()

        if paral_flag:
            model_raw = torch.nn.DataParallel(model_raw)#, device_ids=gpu_ids)
        model_raw = model_raw.to(device)

        test_(model_raw, model_dir, data_dir, s, True)
        res = eval_(model_dir+'/pytorch_result-sp+%.4f.mat'%s)

        res_r.append(res)
    
    
#     print('*reID performace* sparsity =',sparsity, 
#           'rank@1 acc =',[r['acc'][0].cpu().item() for r in res_r], 
#           'mAP =',[r['mAP'] for r in res_r])
