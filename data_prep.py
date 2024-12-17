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
import torch
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models
import os
import read_files

class prostate_cancer_dataset(Dataset):

    def __init__(self, folder,input1FileList, input2FileList, num_slice,
                 neighbor_num=1):
        assert len(input1FileList) == len(input2FileList), "Input lists should have equal lengths"
        self.folder = folder
        self.input1FileList = input1FileList
        self.input2FileList=input2FileList 
        self.neighbor_num=neighbor_num
        self.num_slice=num_slice
    def __len__(self):
        return len(self.input1FileList)

    def __getitem__(self, imNumber):
        filename_input1_list=read_files.get_neighboring_slices(imNumber,self.input1FileList,neighbor_num=self.neighbor_num)
        img_data_input1=[read_files.preprocessInput1(read_files.imageReaderMat(os.path.join(self.folder,name)).astype('float32')) for name in filename_input1_list]
        filename_input2_list=read_files.get_neighboring_slices(imNumber,self.input2FileList,neighbor_num=self.neighbor_num)
        img_data_input2=[read_files.preprocessInput2(read_files.imageReaderMat(os.path.join(self.folder,name)).astype('float32')) for name in filename_input2_list]
        img_list=img_data_input1+img_data_input2
        img_list=[(torch.from_numpy(img)).unsqueeze(0) for img in img_list]
        img_list=torch.cat(img_list,dim=0)
        
        filename_input1=self.input1FileList[imNumber]
        slice_id=float(filename_input1.split('plane-')[1].split('-')[0])
        slice_prop=float(slice_id)/self.num_slice
        slice_channel=torch.ones(img_list.shape[1:3]).unsqueeze(0)*slice_prop
        img_list=torch.cat([img_list,slice_channel])
        return img_list

      