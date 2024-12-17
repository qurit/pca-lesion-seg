# Licensed under the MIT License.
import torch
import torchvision
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class segmentation_model(nn.Module):
    def __init__(self,model_name,size=192,in_channel=3,num_class=1,out=False):
        super(segmentation_model, self).__init__()  
        if model_name=="fcn_resnet101_coco":
            model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21)
            model.classifier[4]=nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
            model.backbone.conv1=nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if model_name=='deeplabv3_resnet101_coco':
            model=torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21,aux_loss=None)
            model.classifier[4]=nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1))
            model.backbone.conv1=nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if model_name == 'unet_resnet101':
            model = smp.Unet(encoder_name="resnet101",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_resnet152':
            model = smp.Unet(encoder_name="resnet152",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_resnet34':
            model = smp.Unet(encoder_name="resnet34",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
            
        if model_name == 'unet_resnet18':
            model = smp.Unet(encoder_name="resnet18",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_vgg16':
            model = smp.Unet(encoder_name="vgg16",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_densenet121':
            model = smp.Unet(encoder_name="densenet121",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'MAnet':
            model=smp.MAnet(in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'UnetPlusPlus':
            model=smp.UnetPlusPlus(in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_mit_b4':
            model = smp.Unet(encoder_name="mit_b4",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_mit_b0':
            model = smp.Unet(encoder_name="mit_b0",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        self.model=model
        self.out=out
    def forward(self, image):
        if self.out:out_mask=self.model(image)['out']
        else: out_mask=self.model(image)
        return out_mask
