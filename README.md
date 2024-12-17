# Automatic segmentation of prostate cancer metastases in PSMA PET/CT images

<p align="left">
    <img alt="PyPI - License" src="https://img.shields.io/badge/license-MIT-blue" height="18" />
</p>

This project contains a deep neural net model for fully-automated detection and segmentation of metastatic prostate cancer lesions in whole-body PET/CT images (imaged with [18F]DCFPyL).

<br/>
The model architecture, training methodology, and testing results are available in the following publication:

*Xu Y, Klyuzhin I, Harsini S, Ortiz A, Zhang S, Bénard F, Dodhia R, Uribe CF, Rahmim A, Lavista Ferres J. Automatic segmentation of prostate cancer metastases in PSMA PET/CT images using deep neural networks with weighted batch-wise dice loss. Comput Biol Med. 2023 May;158:106882*
https://doi.org/10.1016/j.compbiomed.2023.106882
PMID: 37037147

<br/>
<p align="left">
  <a href="https://www.bccrc.ca/dept/io-programs/qurit/"><img src="https://www.bccrc.ca/dept/io-programs/qurit/sites/qurit/files/FINAL_QURIT_PNG_60.png" height="70"/></a>
</p>

---

## Requirements
To run the code in this repository, you will need to install the conda environment.
```bash
conda env create -f environment.yml
```
# Input format
- The input PET, CT, and Label images should be saved in `data_folder` as Matlab 2D arrays, with one subfolder per subject, and one transverse plane per file:
  - `sub-0001`
    - sub-0001-plane-0001-CT.mat
    - ...
    - sub-0001-plane-0268-CT.mat
    - sub-0001-plane-0001-PET.mat
    - ...
    - sub-0001-plane-0268-PET.mat
    - sub-0001-plane-0001-LAB.mat
    - ...
    - sub-0001-plane-0268-LAB.mat
  - `sub-0002`
    - sub-0002-plane-0001-CT.mat
    - ...
    - sub-0002-plane-0268-CT.mat
    - sub-0002-plane-0001-PET.mat
    - ...
    - sub-0002-plane-0268-PET.mat
    - sub-0002-plane-0001-LAB.mat
    - ...
    - sub-0002-plane-0268-LAB.mat
  - `sub-0003`
  - ...
- Image data in each .mat file should be stored in a variable called "image".
- 1st dimension corresponds to the medial axis (right-to-left), and 2nd dimension corresponds to the antero-posterior axis (front-to-back).
- The 2D image dimensions of all modalities are expected to be 192 x 192, with pixel size 3.64 x 3.64 mm.
- PET pixel intensities should be in SUV units.
- The suffix '-LAB.mat' indicates the ground-truth segmentation mask, which is required only for model training.
- For the ground-truth segmentation mask, the pixel value is 0 for background or normal, and i for the ith lesion, i=1,...n, where n is the total number of annotated lesions of the subject. 


## Usage (inference)
- Model inference 
```bash
conda activate proscancer_work_env
python save_pred_mask.py --trained_model_address 'trained_models/unet_resnet34.pt' --save_folder 'outputs' --data_folder 'data_folder' --model_name 'unet_resnet34'
```
- Visualize output masks by `load_outputs.ipynb`. 

# Model training
- Prepare `data_split/dataSplits.csv` with the following columns: file num,lesion_count and set_index. 
    - The `file_num` is a unique integer representing the subject ID. It has a one-to-one mapping to each subfolder of `data_folder`. For example, `file_num = 1` indicates slices in `data_folder/sub-0001`, and `file_num = 3` indicates slices in `data_folder/sub-0003`.  
    - The `set_index` is 0 for training samples, 1 for val samples and 2 for testing samples. 
    - The table must be sorted by `file_num` from low to high.
- Derive and save lesion proportion for each axial slice. It would be used later in the model training phase. 
    ```bash
      from read_files import *
      df=pd.DataFrame()
      for sub_folder in os.listdir('data_folder'):
          folder = os.path.join('data_folder',sub_folder+'/')
          results=derive_slice_lesion_vol_suv_prop(folder)
          df=pd.concat([results,df])
      df.to_csv('metadata/slice_lesion_vol_prop.csv')
    ```
- Train the model 
```bash
python run_model_train.py PATH_TO_SAVE_FOLDER --neighbor_num 1 --num_epochs 20 --adjust_size --batch_size 32 `
```

## Licence 

This project is licenced under the MIT License.

## How to cite

If you are using this model in your projects, kindly include the following citation:

*Xu Y, Klyuzhin I, Harsini S, Ortiz A, Zhang S, Bénard F, Dodhia R, Uribe CF, Rahmim A, Lavista Ferres J. Automatic segmentation of prostate cancer metastases in PSMA PET/CT images using deep neural networks with weighted batch-wise dice loss. Comput Biol Med. 2023 May;158:106882*

## Acknowledgments

This project was supported by the National Institutes of Health (NIH)/Canadian Institutes of Health Research (CIHR) Quantitative Imaging Network (QIN) grant (137993), the CIHR project grant (PJT-162216), the Mitacs Accelerate grant (IT18063), as well as computational resources and services provided in part by Microsoft and separately by the Vice President Research and Innovation at the University of British Columbia.





  


