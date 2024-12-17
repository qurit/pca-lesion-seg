# Licensed under the MIT License.
import torch.nn as nn
import os
import torch
import pandas as pd
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image


def co_trans_list(img_list,transform,mask1=None,seed=None):
    if not seed: 
        seed = np.random.randint(1000000) 
    for ii in range(len(img_list)):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        img_list[ii]=transform(img_list[ii])
    if mask1: 
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mask1=transform(mask1)
        return img_list,mask1
    else: return img_list


    
def elastic_transform_list_arr(img_list,mask1,alpha=1000, sigma=30, random_state=None):
    img_list,mask1 = [np.asarray(img) for img in img_list], np.asarray(mask1)
    if random_state is None: random_state = np.random.RandomState(None)
    if len(img_list[0].shape) < 3:
        img_list=[img.reshape(img.shape[0], img.shape[1], -1) for img in img_list]
    if len(mask1.shape) < 3:
        mask1 = mask1.reshape(mask1.shape[0], mask1.shape[1], -1)
    img_shape = img_list[0].shape
    dx = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), np.arange(img_shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distorted_img_list=[map_coordinates(img, indices, order=1, mode='reflect') for img in img_list]
    distorted_mask1 = map_coordinates(mask1, indices, order=1, mode='reflect')
    distorted_img_list=[distorted_img.reshape(img_shape) for distorted_img in distorted_img_list]
    distorted_mask1 = distorted_mask1.reshape(mask1.shape)
    return [torch.from_numpy(np.squeeze(distorted_img)) for distorted_img in distorted_img_list],torch.from_numpy(np.squeeze(distorted_mask1))  


def elastic_transform_list(img_list,mask1,alpha=1000, sigma=30, random_state=None):
    img_list,mask1 = [np.asarray(img) for img in img_list], np.asarray(mask1)
    if random_state is None: random_state = np.random.RandomState(None)
    if len(img_list[0].shape) < 3:
        img_list=[img.reshape(img.shape[0], img.shape[1], -1) for img in img_list]
    if len(mask1.shape) < 3:
        mask1 = mask1.reshape(mask1.shape[0], mask1.shape[1], -1)
    img_shape = img_list[0].shape
    dx = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), np.arange(img_shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distorted_img_list=[map_coordinates(img, indices, order=1, mode='reflect') for img in img_list]
    distorted_mask1 = map_coordinates(mask1, indices, order=1, mode='reflect')
    distorted_img_list=[distorted_img.reshape(img_shape) for distorted_img in distorted_img_list]
    distorted_mask1 = distorted_mask1.reshape(mask1.shape)
    return [Image.fromarray(np.squeeze(distorted_img)) for distorted_img in distorted_img_list],Image.fromarray(np.squeeze(distorted_mask1))  

    
class CoTransform_list:
    def __init__(self,size,scale=.8,ratio_per=.25,
                color_brightness=.3,color_contrast=.3,color_saturation=.3,color_hue=.1,
                affine_tf=True,affine_degree=10,affine_shear=10,alpha=1000,sigma=30,elastic_tf=True,crop_tf=True,flip_tf=True,color_tf=True,normalize=False):
        self.size=size
        self.scale=scale
        self.ratio_per=ratio_per
        self.color_brightness=color_brightness
        self.color_contrast=color_contrast
        self.color_saturation=color_saturation
        self.color_hue=color_hue
        self.affine_shear=affine_shear
        self.affine_degree=affine_degree
        self.affine_tf=affine_tf
        self.alpha=alpha
        self.sigma=sigma
        self.elastic_tf=elastic_tf
        self.crop_tf=crop_tf
        self.flip_tf=flip_tf
        self.color_tf=color_tf
        self.normalize=normalize
    def __call__(self, img_list,mask1):
        img_list,mask1=[torch.from_numpy(img) for img in img_list], torch.from_numpy(mask1)
        img_list,mask1=[functional.to_pil_image(img) for img in img_list], functional.to_pil_image(mask1)
        affine = transforms.RandomAffine(degrees=self.affine_degree,shear=self.affine_shear)
        if self.affine_tf: img_list,mask1=co_trans_list(img_list,transform=affine,mask1=mask1)
        crop = transforms.RandomResizedCrop(self.size,scale=(self.scale, 1.0),ratio=(1-self.ratio_per, 1+self.ratio_per)) # scaling is something need more attention. Should not scale by a lot. 
        if self.crop_tf: img_list,mask1=co_trans_list(img_list,transform=crop,mask1=mask1)
        if self.flip_tf: 
            img_list,mask1=co_trans_list(img_list,mask1=mask1,transform=transforms.RandomHorizontalFlip())
            img_list,mask1=co_trans_list(img_list,mask1=mask1,transform=transforms.RandomVerticalFlip())
        color_aug=transforms.ColorJitter(self.color_brightness,self.color_contrast,self.color_saturation,self.color_hue) # difference matters
        if self.color_tf: img_list=co_trans_list(img_list,transform=color_aug)
        if self.elastic_tf: img_list,mask1=elastic_transform_list(img_list,mask1,alpha=self.alpha, sigma=self.sigma)
        img_list,mask1=[functional.to_tensor(img) for img in img_list],transforms.functional.to_tensor(mask1)
        img_list=torch.cat(img_list,dim=0)
        if self.normalize: img_list=[functional.normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in img_list]
        
        mask1=(mask1>0).int()
        return img_list,mask1

class CoTransform_list0:
    def __init__(self,size,scale=.8,ratio_per=.25,
                alpha=1000,sigma=30,elastic_tf=True,
                crop_tf=True,flip_tf=True,normalize=False):
        self.size=size
        self.scale=scale
        self.ratio_per=ratio_per
        self.alpha=alpha
        self.sigma=sigma
        self.elastic_tf=elastic_tf
        self.crop_tf=crop_tf
        self.flip_tf=flip_tf
        self.normalize=normalize
    def __call__(self, img_list,mask1):
        img_list,mask1=[torch.from_numpy(img) for img in img_list], torch.from_numpy(mask1)
        img_list=[torch.unsqueeze(img, dim=0) for img in img_list]
        mask1=torch.unsqueeze(mask1, dim=0)
        crop = transforms.RandomResizedCrop(self.size,scale=(self.scale, 1.0),ratio=(1-self.ratio_per, 1+self.ratio_per)) # scaling is something need more attention. Should not scale by a lot. 
        if self.crop_tf: img_list,mask1=co_trans_list(img_list,transform=crop,mask1=mask1)
        if self.flip_tf: 
            img_list,mask1=co_trans_list(img_list,mask1=mask1,transform=transforms.RandomHorizontalFlip())
            img_list,mask1=co_trans_list(img_list,mask1=mask1,transform=transforms.RandomVerticalFlip())
        mask1=torch.squeeze(mask1)
        img_list=[torch.squeeze(img) for img in img_list]
        if self.elastic_tf: img_list,mask1=elastic_transform_list_arr(img_list,mask1,alpha=self.alpha, sigma=self.sigma)
        img_list=[img.unsqueeze(0) for img in img_list]
        img_list=torch.cat(img_list,dim=0)
        if self.normalize: img_list=[functional.normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in img_list]
        
        mask1=(mask1>0).int()
        return img_list,mask1

    
class noTransform_list:
    def __init__(self):
        super(noTransform_list, self).__init__()
    def __call__(self, img_list,mask1):
        img_list,mask1=[torch.from_numpy(img) for img in img_list], torch.from_numpy(mask1)
        img_list,mask1=[functional.to_pil_image(img) for img in img_list], functional.to_pil_image(mask1)
        img_list,mask1=[functional.to_tensor(img) for img in img_list],transforms.functional.to_tensor(mask1)
        mask1=(mask1>0).int()
        return img_list,mask1
class noTransform_list0:
    def __init__(self):
        super(noTransform_list0, self).__init__()
    def __call__(self, img_list,mask1):
        img_list,mask1=[(torch.from_numpy(img)).unsqueeze(0) for img in img_list], torch.from_numpy(mask1)
        img_list=torch.cat(img_list,dim=0)
        mask1=(mask1>0).int()
        return img_list,mask1
    
class normalizeTransform_list:
    def __init__(self):
        super(noTransform_list0, self).__init__()
    def __call__(self, img_list,mask1):
        img_list,mask1=[(torch.from_numpy(img)).unsqueeze(0) for img in img_list], torch.from_numpy(mask1)
        img_list=torch.cat(img_list,dim=0)
        img_list=[functional.normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in img_list]
        mask1=(mask1>0).int()
        return img_list,mask1
