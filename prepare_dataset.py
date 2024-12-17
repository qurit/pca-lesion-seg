# Licensed under the MIT License.
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from read_files import *
import numpy as np
from torchvision import datasets, models
from tf_neigboring import *

class BCCDataset_neighboring(Dataset):
    def __init__(self, folder, input1FileList, input2FileList, outputFileList, 
                 class_weights,
                 transform=None,lesion_count=None,num_slice=None,
                 slice_vol=None,vol_const=None,
                 keep_original_output=False,neighbor_num=1,normalize_pet_max=False):
        assert len(input1FileList) == len(input2FileList), "Input lists should have equal lengths"
        assert len(outputFileList) == len(input1FileList), "Input and output lists should have equal lengths"
        self.folder = folder
        self.input1FileList = input1FileList
        self.input2FileList=input2FileList
        self.outputFileList=outputFileList  
        if transform is None: self.transform = noTransform_list0()
        elif transform == 'normalize': self.transform = normalizeTransform_list()
        else: self.transform = transform
        self.class_weights=class_weights
        self.lesion_count=lesion_count
        self.num_slice=num_slice
        self.slice_vol=slice_vol
        self.const=vol_const
        self.neighbor_num=neighbor_num
        self.normalize_pet_max=normalize_pet_max
        self.keep_original_output=keep_original_output
    def __len__(self):
        return len(self.outputFileList)

    def __getitem__(self, imNumber):
        
        filename_input1_list=get_neighboring_slices(imNumber,self.input1FileList,neighbor_num=self.neighbor_num)
        img_data_input1=[preprocessInput1(imageReaderMat(self.folder+name).astype('float32')) for name in filename_input1_list]
        filename_input2_list=get_neighboring_slices(imNumber,self.input2FileList,neighbor_num=self.neighbor_num)
        if not self.normalize_pet_max: img_data_input2=[preprocessInput2(imageReaderMat(self.folder+name).astype('float32')) for name in filename_input2_list]
        else: img_data_input2=[preprocessInput2max(imageReaderMat(self.folder+name).astype('float32')) for name in filename_input2_list]
        filename_output = self.outputFileList[imNumber]
        img_list=img_data_input1+img_data_input2
        # read output
        img_data_output = imageReaderMat(self.folder+filename_output).astype('float32')
        if transform or not self.keep_original_output: img_data_output = preprocessOutput(img_data_output)
        if img_data_output.sum()>0: 
            weight=self.class_weights[1]
            with_lesion=1
        else: 
            weight=self.class_weights[0]
            with_lesion=0
        img_list,img_data_output=self.transform(img_list,img_data_output)
        if self.lesion_count is None: return img_list, img_data_output, weight, with_lesion
        else: 
            filename_input1=self.input1FileList[imNumber]
            file_num=int(filename_input1.split('/')[0].split('-')[1])
            num_lesion_patient=self.lesion_count[file_num-1]
            if self.num_slice is None: return img_list, img_data_output, weight, with_lesion,num_lesion_patient
            else:
                slice_id=float(filename_input1.split('plane-')[1].split('-')[0])
                slice_prop=float(slice_id)/self.num_slice[file_num-1]
                slice_channel=torch.ones(img_list.shape[1:3]).unsqueeze(0)*slice_prop
                img_list=torch.cat([img_list,slice_channel])
                if self.slice_vol is None or self.const is None:
                    return img_list, img_data_output, weight, with_lesion,num_lesion_patient
                else: 
                    lesion_vol=float(self.slice_vol.loc[self.slice_vol.filename_output==filename_output].lesion_vol.values[0])
                    if lesion_vol==0: tp_ratio=1.0
                    else: tp_ratio=self.const/lesion_vol
                    return img_list, img_data_output, weight, with_lesion,num_lesion_patient,tp_ratio
                
          
