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


def run_segmentation_model_batch_adapt_neighboring_adjustSize(model, 
                             train_loader, 
                             val_lists,
                             criterion_seg,
                             optimizer, 
                             data_folder,
                             class_weights,
                             lesion_count,
                             slice_vol,
                             vol_const,
                             neighbor_num=1,
                             num_slice_p=None,
                             save_dir=None,
                             num_epochs=25,
                             skip_all_neg_batch=False,
                            normalize_pet_max=False,
                             adjust_size=True,
                             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                             bs=32):
    since = time.time()
    segmentation_history={'train':{'loss':[],'iou':[],'sen_lesion_slice':[],'spec_no_lesion_slice':[]},
                           'val': {'iou':[],'loss':[],'sen_lesion_slice':[],'spec_no_lesion_slice':[]}
                         }
    best_model = copy.deepcopy(model.state_dict())
    best_model_loss = copy.deepcopy(model.state_dict())
    best_iou,best_loss=0,100.0
    if save_dir:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase='train'
        model.train()  
        running_loss_s,loss_lesion,num_lesion_slice = 0.0,0.0,0.0
        running_loss_c,TN,FP,FN,TP=.0,.0,.0,.0,.0
        sen_lesion_slice,spec_no_lesion_slice=0.0,0.0
        intersect,union=0.0,0.0
        for inputs, labels, weights, with_lesion, num_lesion, tp_ratios in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights=weights.to(device).float()
            with_lesion=with_lesion.to(device).float()
            num_lesion=num_lesion.to(device)
            tp_ratios=tp_ratios.to(device).float()
            out_masks = model(inputs)
            if adjust_size: loss1,a,b=criterion_seg(out_masks,labels,num_lesion,tp_ratios)
            else: loss1,a,b=criterion_seg(out_masks,labels,num_lesion,None)
            optimizer.zero_grad()
            loss1.backward()
            if not skip_all_neg_batch or with_lesion.sum().item()>0: optimizer.step()
            else: optimizer.zero_grad()
            intersect += a.item()
            union += b.item()
            running_loss_s += loss1.mean().item() * inputs.size(0)
            num_lesion_slice += with_lesion.sum().item()
            pred_mask=(out_masks>0).int()
            pred_mask=pred_mask.view(pred_mask.size(0),-1)
            labels_flatten=labels.view(labels.size(0),-1)
            sen_lesion_slice+=((pred_mask*labels_flatten).sum(1).float()/(labels_flatten.sum(1).float()+1e-9)*with_lesion).sum().item()
            spec_no_lesion_slice+=(((1-pred_mask)*(1-labels_flatten)).sum(1).float()/((1-labels_flatten).sum(1).float()+1e-9)*(1-with_lesion)).sum().item()
        
        epoch_loss_s = running_loss_s / len(train_loader.dataset)
        iou_train = intersect/union
        sen_lesion_slice = sen_lesion_slice/num_lesion_slice
        spec_no_lesion_slice = spec_no_lesion_slice/(len(train_loader.dataset)-num_lesion_slice)
        print('Train Loss: {:.4f} IOU: {:.4f} sen Lesion Slice: {:.4f} spec No Lesion Slice: {:.4f}'.format(epoch_loss_s,iou_train,sen_lesion_slice,spec_no_lesion_slice))
        segmentation_history[phase]['loss'].append(epoch_loss_s)
        segmentation_history[phase]['iou'].append(iou_train)
        segmentation_history[phase]['sen_lesion_slice'].append(sen_lesion_slice)
        segmentation_history[phase]['spec_no_lesion_slice'].append(spec_no_lesion_slice)
        
        phase='val'
        model.eval()   
        with torch.no_grad():
            iou=0.0
            num_slice =0.0
            running_loss_s,loss_lesion,num_lesion_slice = 0.0,0.0,0.0
            sen_lesion_slice,spec_no_lesion_slice=0.0,0.0
            for kk in range(len(val_lists['inputList1'])):
                intersection,union,tumors=0.0,0.0,0.0
                val_data=BCCDataset_neighboring(data_folder, 
                                    val_lists['inputList1'][kk],
                                    val_lists['inputList2'][kk],
                                    val_lists['outputList'][kk],
                                    transform=None,
                                    neighbor_num=neighbor_num,
                                    class_weights=class_weights,
                                     lesion_count=lesion_count,
                                     num_slice=num_slice_p, normalize_pet_max=normalize_pet_max,
                                    slice_vol=slice_vol,vol_const=vol_const
                                   )
                val_loader=DataLoader(val_data, batch_size=bs,shuffle=False, num_workers=4)                
                for inputs, labels, weights,with_lesion,num_lesion,tp_ratios in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    weights=weights.to(device).float()
                    with_lesion=with_lesion.to(device).float()
                    num_lesion=num_lesion.to(device)
                    tp_ratios=tp_ratios.to(device).float()
                    out_masks = model(inputs)
                    if adjust_size: loss1,_,_=criterion_seg(out_masks,labels,num_lesion,tp_ratios)
                    else: loss1,_,_=criterion_seg(out_masks,labels,num_lesion,None)
                    running_loss_s += loss1.mean().item() * inputs.size(0)
                    num_lesion_slice += with_lesion.sum().item()
                    num_slice +=inputs.size(0)
                    pred_mask=(out_masks>0).int()
                    pred_mask=pred_mask.view(pred_mask.size(0),-1)
                    labels_flatten=labels.view(labels.size(0),-1)
                    sen_lesion_slice+=((pred_mask*labels_flatten).sum(1).float()/(labels_flatten.sum(1).float()+1e-9)*with_lesion).sum().item()
                    spec_no_lesion_slice+=(((1-pred_mask)*(1-labels_flatten)).sum(1).float()/((1-labels_flatten).sum(1).float()+1e-9)*(1-with_lesion)).sum().item()
                    inter,uni,tt = get_intersection_union(out_masks, labels)
                    intersection += inter
                    union += uni
                    tumors += tt
                iou+=intersection/union  
            

            iou=iou/(len(val_lists['inputList1']))
            

            epoch_loss_s = running_loss_s / num_slice
            sen_lesion_slice = sen_lesion_slice/num_lesion_slice
            spec_no_lesion_slice = spec_no_lesion_slice/(num_slice-num_lesion_slice)
            print('Val IoU {:.4f} Loss: {:.4f} sen Lesion Slice: {:.4f} spec No Lesion Slice: {:.4f}'.format(iou,epoch_loss_s,sen_lesion_slice,spec_no_lesion_slice))

            segmentation_history[phase]['iou'].append(iou)
            segmentation_history[phase]['loss'].append(epoch_loss_s)
            segmentation_history[phase]['sen_lesion_slice'].append(sen_lesion_slice)
            segmentation_history[phase]['spec_no_lesion_slice'].append(spec_no_lesion_slice)
        if(epoch_loss_s<best_loss):
            best_loss=epoch_loss_s
            best_model_loss=copy.deepcopy(model.state_dict())
            torch.save(best_model_loss, os.path.join(save_dir,"best_model_loss.pt"))
        if(iou>best_iou):
            best_iou=iou
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(save_dir,"best_model.pt"))
        if save_dir:
            if epoch%2==0:
                torch.save(model.state_dict(), os.path.join(save_dir,'model'+str(epoch)+'.pt'))
                
                pd.DataFrame(segmentation_history['train']).to_csv(os.path.join(save_dir,'train_segmentation.csv'),index=False)
                pd.DataFrame(segmentation_history['val']).to_csv(os.path.join(save_dir,'val_segmentation.csv'),index=False)
                      
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val iou: {:4f}'.format(best_iou))
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir,'model.pt'))
        torch.save(best_model, os.path.join(save_dir,"best_model.pt"))
        torch.save(best_model_loss, os.path.join(save_dir,"best_model_loss.pt"))
        pd.DataFrame(segmentation_history['train']).to_csv(os.path.join(save_dir,'train_segmentation.csv'),index=False)
        pd.DataFrame(segmentation_history['val']).to_csv(os.path.join(save_dir,'val_segmentation.csv'),index=False)
    return model,best_model,best_model_loss,segmentation_history


