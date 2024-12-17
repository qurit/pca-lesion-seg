# Licensed under the MIT License.
import scipy
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
import numpy as np
import torchvision
from torchvision import datasets, models
from models import *
import data_prep
import argparse

def save_tumor_mask(model,data_folder,address_plane_CT_slices,address_plane_PET_slices,neighbor_num=1,save_folder='outputs'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_folder):os.makedirs(save_folder)
    model.eval()   
    with torch.no_grad():
        patient_sub=address_plane_CT_slices[0].split('/')[0]
        num_slice=len(address_plane_CT_slices)
        patient_data=data_prep.prostate_cancer_dataset(
                 data_folder, 
                 address_plane_CT_slices,
                    address_plane_PET_slices,
                    neighbor_num=neighbor_num,
                    num_slice=num_slice
                     )
        start_index=0
        img3D = np.zeros((192, 192, num_slice))
        patient_data_loader=DataLoader(patient_data, 
        batch_size=32,shuffle=False, num_workers=4)                
        for inputs in patient_data_loader:
            inputs = inputs.to(device)
            N=inputs.size(0)
            out_masks = model(inputs)
            img3D[:,:, start_index:( start_index+N)]= torch.sigmoid(out_masks).squeeze().permute(1, 2, 0).cpu().numpy()
            start_index+=N
        binary_mask=(img3D>0.5).astype(int)
        scipy.io.savemat(os.path.join(save_folder,patient_sub+'_predmask.mat'), mdict={'pred_mask': binary_mask})
    return None

def save_all_predictions(trained_model_address,
data_folder,
neighbor_num=1,
model_name='unet_resnet34',
save_folder='outputs'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputList1 = getFileListSubdir(data_folder,'CT.mat')
    inputList2 = getFileListSubdir(data_folder,'PET.mat')
    model=segmentation_model(model_name=model_name,size=192,in_channel=2*(neighbor_num*2+1)+1,num_class=1,out=False)
    model.load_state_dict(torch.load(trained_model_address,map_location=device))
    model=model.to(device)
    for ii in range(len(inputList1)):
        address_plane_CT_slices=inputList1[ii]
        address_plane_PET_slices=inputList2[ii]
        save_tumor_mask(model,
        data_folder=data_folder,
        address_plane_CT_slices=address_plane_CT_slices,
        address_plane_PET_slices=address_plane_PET_slices,
        neighbor_num=neighbor_num,save_folder=save_folder)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_address',type=str, help="address of saved model state_dict")
    parser.add_argument('--save_folder',type=str, help="dir to save outputs")
    parser.add_argument('--data_folder', type=str, help="dir of all the testing patients")
    parser.add_argument('--model_name', type=str, help="model name")
    parser.add_argument('--neighbor_num', type=int, default=1)
    args = parser.parse_args()
    save_all_predictions(args.trained_model_address,
                            args.data_folder,
                            neighbor_num=args.neighbor_num,
                            model_name=args.model_name,
                            save_folder=args.save_folder)

if __name__=="__main__":
    main()
