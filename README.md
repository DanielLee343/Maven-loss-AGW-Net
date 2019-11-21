# Maven loss with Improved Attention W-Net

This repo contains the code of our paper **Maven Loss with Improved Attention W-Net for Biomedical Image Segmentation**.

We introduce a novel loss function named ‘Maven Loss’ by taking ‘specificity’ into consideration to handle the issue of data disequilibrium and to help achieve weighing both abilities of correctly-segmented lesion and non-lesion areas.

### Training
Please create 3 folders before training: `orig_gt`, `resized_train`, `resized_gt`, for full resolution ground truth images, resized images of training set and resized images of ground truth, respectively. The resized resolution is 192*256.

### Citation 

This work is based on the work blow:
https://github.com/nabsabraham/focal-tversky-unet