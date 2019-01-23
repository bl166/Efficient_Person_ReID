from __future__ import absolute_import
import math
import torch
import torch.nn as nn

if __name__ == '__main__':
    import os,sys; sys.path.append(os.getcwd())
    from channel_selection import channel_selection
    from model_legacy import ClassBlock
else:
    from .channel_selection import channel_selection
    from .model_legacy import ClassBlock    


__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    

class resnet(nn.Module):
    def __init__(self, depth=164, nClass=1000, cfg=None):
#         super(resnet, self).__init__()
        super().__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        self.name='resnet%d'%depth
        
        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
#             cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [[64, 64, 64], [256, 64, 64]*(n-1), [256, 128, 128], [512, 128, 128]*(n-1), [512, 256, 256], [1024, 256, 256]*(n-1), [1024]]
            cfg = [item for sub_list in cfg for item in sub_list]
            
            
        self.argws = {'depth': depth, 'nClass':nClass}
        self.cfg = cfg

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 128, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 256, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(256 * block.expansion)
        self.select = channel_selection(256 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(cfg[-1], nClass)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))#fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
    
# Define the ResNet50-based Model
class ft_resnet(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = resnet(47)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(self.model.cfg[-1], class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.layer1(x)  # 32x32
        x = self.model.layer2(x)  # 16x16
        x = self.model.layer3(x)  # 8x8
        x = self.model.bn(x)
        x = self.model.select(x)
        x = self.model.relu(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        
        x = self.classifier(x)
        return x
    

    
if __name__ == "__main__":

    import torch 
    from torch.autograd import Variable
    from prune import prune 

    net = ft_resnet(751)
    print(net.model.cfg)

    final_net = ft_resnet(751)
    final_net.model = prune(net.model, 0.5, False)
    print(final_net)

    input = Variable(torch.FloatTensor(4, 3, 128, 256)).cuda()
    
    net.cuda()
    final_net.cuda()

    print('\n******** validating pruned forward path ********')
    print('final output size:',final_net.model(input).shape)

    print(len([l for l in final_net.state_dict().keys() if 'conv' in l.lower()]))
