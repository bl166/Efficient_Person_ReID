import scipy.io
import numpy as np
import time
import os
import warnings

import torch
from torch.autograd import Variable

try:
    from func import *
except:
    from .func import *
    
if is_interactive():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
  
    
######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders,architec,device="cuda"):
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
#         print(count, end='\r')

        if architec.lower() == 'den':
            ff = torch.FloatTensor(n,1024).zero_()
        elif architec.lower() == 'res':
            ff = torch.FloatTensor(n,2048).zero_()
        elif architec.lower() == 'pcb':
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        else:
            raise ValueError
            
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.to(device))
            outputs = model(input_img) 
            f = outputs.data.cpu()
            ff += f
            
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
        
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels



#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc,use_gpu=True):
    if use_gpu:
        query = qf.view(-1,1)
        # print(query.shape)
        score = torch.mm(gf,query).squeeze(1).cpu().numpy()
    else:
        query = qf
        score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


######################################################################
    
def main(device="cuda"):
    
    cwd = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    result = scipy.io.loadmat('%s/.tmp/pytorch_result.mat'%cwd)
    
    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    
    gallery_feature = result['gallery_f']        
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
        
    query_feature = torch.FloatTensor(query_feature).to(device)
    gallery_feature = torch.FloatTensor(gallery_feature).to(device)

    multi = os.path.isfile('%s/.tmp/multi_query.mat'%cwd)
    if multi:
        m_result = scipy.io.loadmat('%s/.tmp/multi_query.mat'%cwd)
        mquery_feature = m_result['mquery_f']
        mquery_cam = m_result['mquery_cam'][0]
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = torch.FloatTensor(mquery_feature).to(device)

#     print(query_feature.shape)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    #print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],
                                   gallery_feature,gallery_label,gallery_cam, device=="cuda")
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

    # multiple-query
    if multi:
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for i in range(len(query_label)):
            mquery_index1 = np.argwhere(mquery_label==query_label[i])
            mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
            mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
            mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
            ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])
        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
        
    return CMC,ap/len(query_label)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device)