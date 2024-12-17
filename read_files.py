# Licensed under the MIT License.
import os
import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

def getFileListSubdir(folder,suffix):
    # list of lists containing file names with specific ending
    subFolderList = os.listdir(folder)
    subFolderList.sort()
    fileList = list()
    for subFolder in subFolderList:
        fileListSubfolder = [subFolder+'/'+file for file in os.listdir(os.path.join(folder,subFolder)) if file.endswith(suffix)]
        fileListSubfolder.sort()
        fileList.append(fileListSubfolder)
    return fileList

def get_files(folder,suffix):
    return [f for f in os.listdir(folder) if f.endswith(suffix)]

def imageReaderMat(filenameFull):
    # reading n-d data from matlab file
    varName = 'image' # name of struct/dict field
    FA_org = sio.loadmat(filenameFull)
    img_data = FA_org[varName]
    return img_data


def imageWriterMat(filenameFull,imageData):
    # write n-d image to a matlab file
    filenameDir = os.path.split(filenameFull)[0] + '/'
    if not os.path.exists(filenameDir):
        os.mkdir(filenameDir)
    varName = 'image' # name of struct/dict field
    mdict = {varName:imageData}
    sio.savemat(filenameFull,mdict)


# This enhances the contrast of soft tissues

def preprocessInput1(data):
    # pre-process channel 1 image - CT
    data[data>1200]=1200
    data = data-800 
    data[data<0]=0
    data = data/400
    return data

def preprocessInput2(data,trunc=50):
    # pre-process channel 2 image - PET
    data[data<0] = 0
    data[data>trunc]=trunc
    data=data/trunc
    return data

def preprocessInput2max(data,trunc=50):
    # pre-process channel 2 image - PET
    data[data<0] = 0
    data[data>trunc]=trunc
    data=data/data.max()
    return data

def preprocessOutput(data):
    # pre-process output - convert all integer labels to 1
    data[data>1] = 1
    return data

def get_neighboring_slices(imNumber,inputList_flat,neighbor_num):
    filename_input=inputList_flat[imNumber] # the center slice
    start_index,end_index=imNumber-neighbor_num,imNumber+neighbor_num
    slices=[]
    before_center_slice=True
    num_slice_from_prev_patient=0
    slice_index=start_index
    # count neighboring slices belong to the previous patient, replace these slices with the first slice from the current patient
    while slice_index<imNumber and inputList_flat[slice_index].split('/')[0]!=filename_input.split('/')[0]:
        slice_index+=1
        num_slice_from_prev_patient+=1
    if num_slice_from_prev_patient>0: slices=[inputList_flat[slice_index]]*num_slice_from_prev_patient
    while slice_index<min(end_index+1,len(inputList_flat)) and inputList_flat[slice_index].split('/')[0]==filename_input.split('/')[0]:
        slices.append(inputList_flat[slice_index])
        slice_index+=1
    # If there are neighboring slices from the next patient, replace these with the last slice from the current patient. 
    if slice_index!=end_index+1:
        num_slice_from_nex_patient=end_index+1-slice_index
        slices+=[inputList_flat[slice_index-1]]*num_slice_from_nex_patient
    return slices
    
def readVolume(folder, suffix):
    # list of lists containing file names with specific ending
    fileList = [file for file in os.listdir(folder) if file.endswith(suffix)]
    fileList.sort()
    img2D = imageReaderMat(os.path.join(folder,fileList[0]))
    img3D = np.zeros((img2D.shape[0], img2D.shape[1], len(fileList)))
    for idx,file in enumerate(fileList):
        img2D = imageReaderMat(os.path.join(folder,file))
        img3D[:,:,idx] = img2D
    return img3D


def MIP(img3D,axisNum=1):
    # plot maximum intensity projection of a 3D image
    img2D = np.max(img3D, axis=axisNum)
    if axisNum==1:
        img2D = np.rot90(img2D,k=-1)
    return img2D

'''
example:
subjectNum=10
subjectStr = str(subjectNum).zfill(4)
folder = os.path.join('data_folder','sub-'+subjectStr)
derive_slice_lesion_vol(folder)
'''
def derive_slice_lesion_vol_suv_prop(folder):
    fileList = [file for file in os.listdir(folder) if file.endswith('LAB.mat')]
    fileList.sort()
    sub_folder=fileList[0].split('-plane')[0]
    imgLABEL = readVolume(folder,'LAB.mat')
    imgPET = readVolume(folder,'PET.mat')
    lesions=np.unique(imgLABEL)
    lesions=lesions[lesions>0]
    lesion_volume={key: (imgLABEL==key).sum() for key in lesions}
    lesion_intensity_mean={key: imgPET[imgLABEL==key].mean() for key in lesions}
    lesion_intensity_max={key: imgPET[imgLABEL==key].max() for key in lesions}
    slice_lesion_vol=[]
    slice_suv_mean=[]
    slice_suv_max=[]
    slice_lesion_prop=[]
    for slice_index in range(imgLABEL.shape[2]):
        img_slice=imgLABEL[:,:,slice_index]
        lesion=np.unique(img_slice)
        lesion=lesion[lesion>0]
        if len(lesion)==0: 
            slice_lesion_vol.append(0)
            slice_suv_mean.append(0)
            slice_suv_max.append(0)
            slice_lesion_prop.append(0)
        elif len(lesion)==1: 
            slice_lesion_vol.append(lesion_volume[lesion[0]])
            slice_suv_mean.append(lesion_intensity_mean[lesion[0]])
            slice_suv_max.append(lesion_intensity_max[lesion[0]])
            lesion_pixel_cnt=(img_slice==lesion[0]).sum()*1.0
            slice_lesion_prop.append(lesion_pixel_cnt/lesion_volume[lesion[0]])
        else: 
            vols=[lesion_volume[x] for x in lesion]
            suv_avgs=[lesion_intensity_mean[x] for x in lesion]
            suv_maxs=[lesion_intensity_max[x] for x in lesion]
            lesion_pixel_cnts=[(img_slice==x).sum() for x in lesion]
            lesion_vol_prop=[x/y for (x,y) in zip(lesion_pixel_cnts,vols)]
            
            lesion_index=lesion_vol_prop.index(max(lesion_vol_prop))
            slice_lesion_vol.append(vols[lesion_index])
            slice_suv_mean.append(suv_avgs[lesion_index])
            slice_suv_max.append(suv_maxs[lesion_index])
            slice_lesion_prop.append(lesion_vol_prop[lesion_index])
                
    return pd.DataFrame({'filename_output':[sub_folder+'/'+xx for xx in fileList],
                         'lesion_vol':slice_lesion_vol,
                        'suv_mean':slice_suv_mean,
                        'suv_max':slice_suv_max,
                        'lesion_prop': slice_lesion_prop})
