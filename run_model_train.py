# Licensed under the MIT License.
import sys
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
import torch
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
from prepare_dataset import *
from tf_neigboring import *
from loss import *
from torch.utils.data import DataLoader
from train_segmentation import *
from models import *
import argparse


def write_file(filename, **kwargs):
    with open(filename, "w") as handle:
        for key, value in kwargs.items():
            handle.write("{}: {}\n" .format(key, value))
            
parser = argparse.ArgumentParser()
parser.add_argument('save_dir', help="directory to save experiment results")
parser.add_argument("--DATA_FOLDER", default='data_folder')
parser.add_argument( "--cudaID", default=0,type=int,help="cuda ID")
parser.add_argument( "--batch_size", default=32,type=int,help="batch size")
parser.add_argument( "--num_epochs", default=50,type=int)
parser.add_argument( "--neighbor_num", default=1,type=int)
parser.add_argument( "--model_name", default='unet_resnet34')
parser.add_argument( "--adjust_size", action='store_true')
parser.add_argument( "--optim_method", default='SGD')
parser.add_argument( "--lr", default=0.003,type=float)
parser.add_argument( "--no_train_tf", action='store_true')

args = parser.parse_args()

device=torch.device('cuda:'+str(args.cudaID))
if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
save_dir=args.save_dir

batch_size=args.batch_size
num_epochs=args.num_epochs
num_workers=4
neighbor_num=args.neighbor_num

DATA_FOLDER = args.DATA_FOLDER+'/'
INPUT_SHAPE = (192,192, 2) 
OUTPUT_SHAPE = (192,192,1)

inputList1 = getFileListSubdir(DATA_FOLDER,'CT.mat')
inputList2 = getFileListSubdir(DATA_FOLDER,'PET.mat')
outputList = getFileListSubdir(DATA_FOLDER,'LAB.mat')
data_split=pd.read_csv('data_split/dataSplits.csv')
train_files=data_split.loc[data_split['set_index']==0,'file num']
train_indexes=[x-1 for x in train_files]
val_files=data_split.loc[data_split['set_index']==1,'file num']
val_indexes=[x-1 for x in val_files]
test_files=data_split.loc[data_split['set_index']==2,'file num']
test_indexes=[x-1 for x in test_files]

lesion_count=data_split.lesion_count.tolist()

inputList1_train, inputList2_train, outputList_train  = [inputList1[ii] for ii in train_indexes], [inputList2[ii] for ii in train_indexes], [outputList[ii] for ii in train_indexes]
inputList1_val,inputList2_val,outputList_val = [inputList1[ii] for ii in val_indexes], [inputList2[ii] for ii in val_indexes],[outputList[ii] for ii in val_indexes]
inputList1_test,inputList2_test,outputList_test= [inputList1[ii] for ii in test_indexes],[inputList2[ii] for ii in test_indexes],[outputList[ii] for ii in test_indexes]


inputList1_train_flat = [item for sublist in inputList1_train for item in sublist]
inputList2_train_flat = [item for sublist in inputList2_train for item in sublist]
outputList_train_flat = [item for sublist in outputList_train for item in sublist]

inputList1_val_flat = [item for sublist in inputList1_val for item in sublist]
inputList2_val_flat = [item for sublist in inputList2_val for item in sublist]
outputList_val_flat = [item for sublist in outputList_val for item in sublist]

inputList1_test_flat = [item for sublist in inputList1_test for item in sublist]
inputList2_test_flat = [item for sublist in inputList2_test for item in sublist]
outputList_test_flat = [item for sublist in outputList_test for item in sublist]

num_slice=[len(xx) for xx in inputList1]
slice_vol=pd.read_csv('metadata/slice_lesion_vol_prop.csv')
slice_vol_train=slice_vol.loc[slice_vol.filename_output.isin(outputList_train_flat)]
slice_vol_train_nonzero=slice_vol_train[slice_vol_train.lesion_vol>0]
mean_vol=slice_vol_train_nonzero.lesion_vol.mean()
median_vol=slice_vol_train_nonzero.lesion_vol.median()


total,exist_lesion=0,0
contain_lesion=[]
for filename_output in outputList_train_flat:
    img_data_output = imageReaderMat(os.path.join(DATA_FOLDER,filename_output))
    img_data_output = img_data_output.astype('float32')
    if img_data_output.sum()>0:
        exist_lesion+=1
        contain_lesion.append(True)
    else:
        contain_lesion.append(False)
    total+=1
    
total=len(contain_lesion)
exist_lesion=sum(contain_lesion)

class_weights=[total/(total-exist_lesion),total/exist_lesion]
train_weights=[class_weights[int(label)] for label in contain_lesion]
sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(contain_lesion),replacement=True)
train_transform=CoTransform_list(size=192)
if args.no_train_tf:train_transform=None
model=segmentation_model(model_name=args.model_name,size=192,in_channel=2*(neighbor_num*2+1)+1,num_class=1,out=False)
prev_model_path=os.path.join(args.save_dir,'model.pt')
if os.path.isfile(prev_model_path): 
    model.load_state_dict(torch.load(prev_model_path,map_location=device))
    print(f'Loaded model from {prev_model_path}')
model=model.to(device)
wd=0
if args.optim_method=='SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,weight_decay=wd)
else: optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=wd)
criterion_seg=TverskyLossBatchAdjustSize()
train_data=BCCDataset_neighboring(
                 DATA_FOLDER, 
                 inputList1_train_flat,
                 inputList2_train_flat,
                 outputList_train_flat,
                 transform=train_transform,
                    class_weights=class_weights,
                    lesion_count=lesion_count,
    num_slice=num_slice,
    slice_vol=slice_vol,
    vol_const=median_vol,
    neighbor_num=neighbor_num
                     )

val_lists={'inputList1':inputList1_val,
             'inputList2':inputList2_val,
                'outputList': outputList_val
}
train_loader=DataLoader(train_data,batch_size=batch_size,num_workers=4,sampler=sampler)
results=run_segmentation_model_batch_adapt_neighboring_adjustSize(model, 
                             train_loader, 
                             val_lists,
                             criterion_seg,
                             optimizer, 
                             DATA_FOLDER,
                             class_weights,
                            lesion_count,
                            slice_vol,                                                            
                             median_vol,
                            save_dir=save_dir,
                             num_epochs=num_epochs,
                            num_slice_p=num_slice,
                            neighbor_num =neighbor_num,
                             device =device,
                             bs=32,
                             adjust_size=args.adjust_size)

write_file(os.path.join(args.save_dir,'parameters.txt'), 
           batch_size=args.batch_size,
           num_epochs=args.num_epochs,
           model_name=args.model_name,
           adjust_size=args.adjust_size,
           optim_method=args.optim_method)