#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 08:09:47 2021

@author: tayfunkaraderi
"""

# PyTorch stuff ------
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
    

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'

}

# 3x3 convolution with padding
def conv3x3(in_planes, out_planes, stride=1, groups = 32):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups = 32,  bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1): #-> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=4):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=4):
        super(Bottleneck, self).__init__()
        
        width = int(planes * (4 / 64.)) * 16 ##
        print(width)
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups = 32, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 2 , kernel_size=1, bias=False) # * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)#self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #self.downsample = downsample # conv1x1(inplanes, planes, stride=1)
        self.stride = stride
        

    def forward(self, x, downsample=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #if self.downsample is not None:
        #    residual = self.downsample(x)
        
        #print('out', out.shape)
        #print('res', x.shape)
        
        if out.shape[1] != x.shape[1]:
            if out.shape[1] == 4 * x.shape[1]:
                residual = x.repeat(1, 4, 1, 1)
                #print('res4x', residual.shape)
                
            if out.shape[1] == 2 * x.shape[1]:
                residual = x.repeat(1, 2, 1, 1)
                #print('res2x', residual.shape)

                residual = nn.MaxPool2d(2, stride=2)(residual)
                #print('res2x_pool', residual.shape)
            
            
                
        #if out.shape[1] == 4 * x.shape[1]:
        #    residual = x.repeat(1, 4, 1, 1)
                #print('res4x', residual.shape)
                
        #else: #out.shape[1] == 2 * x.shape[1]:
        #    residual = x.repeat(1, 2, 1, 1)
                #print('res2x', residual.shape)
        #    residual = nn.MaxPool2d(2, stride=2)(residual)

        
        out += residual
        out = self.relu(out)

        return out

class Triplet_ResNet_Softmax(nn.Module):
    def __init__(self, block, layers, num_classes=35, groups=32, width_per_group=4): ##
        
        self.groups = 32 ##
        self.base_width = 4 ##width_per_group ##
        self.width_per_group = 4 ##

        
        self.inplanes = 64
        super(Triplet_ResNet_Softmax, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  ##was 64 for resnet to 128 for resnext
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, 1000)
        self.fc_softmax = nn.Linear(1024 * block.expansion, 1000)
        self.fc_embedding = nn.Linear(1024 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): # or   elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, Bottleneck):
        #            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #        elif isinstance(m, BasicBlock):
        #            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)
    '''
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    '''

    def forward_sibling(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.relu(x)

        x_softmax = self.fc_softmax(x)
        x_embedding = self.fc_embedding(x)

        return x_softmax, x_embedding

    def forward(self, input1, input2, input3):
        softmax_vec_1, embedding_vec_1 = self.forward_sibling(input1)
        softmax_vec_2, embedding_vec_2 = self.forward_sibling(input2)
        softmax_vec_3, embedding_vec_3 = self.forward_sibling(input3)

        return embedding_vec_1, embedding_vec_2, embedding_vec_3, torch.cat((softmax_vec_1, softmax_vec_2, softmax_vec_3), 0)

#transfer learning        
#'''
def TripletResnet50Softmax(pretrained=False,  num_classes=35, embedding_size=128, **kwargs):
    model = Triplet_ResNet_Softmax(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

    if pretrained:
        # Get imagenet weights for resnet50
        #weights_imagenet = #model_zoo.load_url(model_urls['resnet50'])

        # Re-name some of the layers to match our network
        #weights_imagenet["fc_softmax.weight"] = weights_imagenet["fc.weight"]
        #weights_imagenet["fc_softmax.bias"] = weights_imagenet["fc.bias"]
        #weights_imagenet["fc_embedding.weight"] = weights_imagenet["fc.weight"]
        #weights_imagenet["fc_embedding.bias"] = weights_imagenet["fc.bias"]

        # Try and load the model state
        #model.load_state_dict(weights_imagenet)
        model = Triplet_ResNet_Softmax(Bottleneck, [3, 4, 6, 3], num_classes=35, **kwargs)
        model.fc = nn.Linear(2048, 1000)
        model.fc_embedding = nn.Linear(1000, embedding_size)
        model.fc_softmax = nn.Linear(1000, num_classes)
        PATH =  "/content/drive/MyDrive/foram-MetricLearningIdentification-master/output/_a_resnext_full_data_rotation_augmented_lr0_001_2/best_model_state.pkl"
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state'])
        model.train()

    #model.fc = nn.Linear(2048, 1000)
    #model.fc_embedding = nn.Linear(1000, embedding_size)
    #model.fc_softmax = nn.Linear(1000, num_classes)

    return model

'''
# Construct resnet50 model, if pretrained (bool) returns a model pre-trained on imagenet
def TripletResnet50Softmax(pretrained=False,  num_classes=35, embedding_size=128, **kwargs): #num_classes=50
    model = Triplet_ResNet_Softmax(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

    if pretrained:
        # Get imagenet weights for resnet50
        weights_imagenet = model_zoo.load_url(model_urls['resnext50_32x4d'])

        # Re-name some of the layers to match our network
        weights_imagenet["fc_softmax.weight"] = weights_imagenet["fc.weight"]
        weights_imagenet["fc_softmax.bias"] = weights_imagenet["fc.bias"]
        weights_imagenet["fc_embedding.weight"] = weights_imagenet["fc.weight"]
        weights_imagenet["fc_embedding.bias"] = weights_imagenet["fc.bias"]

        # Try and load the model state
        model.load_state_dict(weights_imagenet, strict=False)

    model.fc = nn.Linear(2048, 1000)
    model.fc_embedding = nn.Linear(1000, embedding_size)
    model.fc_softmax = nn.Linear(1000, num_classes)

    return model
'''
