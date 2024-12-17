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

class TverskyLossBatchAdjustSize(nn.Module):    
    def __init__(self,smooth = 1E-6,beta=0,alpha=0.5,gamma=1
                ):
        super(TverskyLossBatchAdjustSize, self).__init__()
        self.smooth=smooth
        self.beta=beta
        self.alpha=alpha
        self.gamma=gamma
    def forward(self, pred, ground_truth,num_lesion,tp_ratios=None):
        N = ground_truth.size(0)
        ground_truth = ground_truth.view(N, -1).float()
        ground_truth_TV=ground_truth.sum(1)
        pred=pred.view(N, -1)
        pred = torch.sigmoid(pred)
        loss_TV=torch.mean(torch.abs(pred.sum(1) - ground_truth_TV))
        TP = (pred * ground_truth).sum(1)
        FP = ((1-ground_truth) * pred).sum(1)
        FN = (ground_truth * (1-pred)).sum(1)
        denom=TP + (self.alpha*FP) + (1-self.alpha)*FN
        if tp_ratios is None: 
            loss=((TP).sum() + self.smooth) / ((denom).sum() + self.smooth)  
            return 1-loss+self.beta*loss_TV,TP.sum(),(TP+FP).sum() 
        no_lesion=(tp_ratios==1).float()
        tp_ratios=tp_ratios*(1+(self.gamma-1)*no_lesion)
        loss=((TP*tp_ratios).sum() + self.smooth) / ((denom*tp_ratios).sum() + self.smooth)  
        return 1-loss+self.beta*loss_TV,TP.sum(),(TP+FP+FN).sum()
    
def get_intersection_union(pred, ground_truth):
    N = ground_truth.size(0)
    pred = (pred.view(N, -1)>=0).float()
    ground_truth = ground_truth.view(N, -1).float()
    intersection = (pred * ground_truth).sum().item()
    tumors=ground_truth.sum().item()
    union=tumors + pred.sum().item()-intersection
    return intersection, union, tumors

def jaccard_score(ground_truth,predicted):
    overlap=(ground_truth*predicted).sum()
    return overlap/(ground_truth.sum()+predicted.sum()-overlap)