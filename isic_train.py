# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:16:54 2018

@author: Nabila Abraham
"""

import os 
import cv2 
import numpy as np 
import datetime

import tensorflow as tf
import matplotlib.pyplot as plt 

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization 

#from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

import losses 
import utils 
import newmodels

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
start_whole = datetime.datetime.now()

img_row = 192
img_col = 256
img_chan = 3
epochnum = 50
batchnum = 8
smooth = 1.
input_size = (img_row, img_col, img_chan)
    
sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
#adam = Adam(lr=1e-3) 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#设立文件目录
curr_dir = os.getcwd()
train_dir = os.path.join(curr_dir, 'resized_train')
gt_dir = os.path.join(curr_dir, 'resized_gt')
orig_dir = os.path.join(curr_dir, 'orig_gt')

img_list = os.listdir(train_dir)
num_imgs = len(img_list)

#初始化data
orig_data = np.zeros((num_imgs, img_row, img_col, img_chan))
orig_masks = np.zeros((num_imgs, img_row, img_col,1))

#读取数据
for idx,img_name in enumerate(img_list): 
    try:
        orig_data[idx] = plt.imread(os.path.join(train_dir, img_name))
        orig_masks[idx,:,:,0] =  plt.imread(os.path.join(gt_dir, img_name.split('.')[0] + "_segmentation.png"))
    except Exception as e:
        print('number:',img_name)
indices = np.arange(0,num_imgs,1)

imgs_train, imgs_test, \
imgs_mask_train, orig_imgs_mask_test,\
trainIdx, testIdx = train_test_split(orig_data,orig_masks, indices,test_size=0.25)

imgs_train /= 255
imgs_test /=255

estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='auto')
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_final_dsc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='max')
gt1 = imgs_mask_train[:,::8,::8,:]
gt2 = imgs_mask_train[:,::4,::4,:]
gt3 = imgs_mask_train[:,::2,::2,:]
gt4 = imgs_mask_train
gt_train = [gt1,gt2,gt3,gt4]

model = newmodels.attn_wnet(sgd, input_size, losses.maven_loss)
hist = model.fit(imgs_train, gt_train, validation_split=0.15,
                 shuffle=True, epochs=epochnum, batch_size=batchnum,
                 verbose=True, callbacks=[checkpoint])#, callbacks=[estop,tb])
h = hist.history
utils.plot(h, epochnum, batchnum, img_col, 1)

num_test = len(imgs_test)
_,_,_,preds = model.predict(imgs_test)
#preds = model.predict(imgs_test)   #use this if the model is unet

preds_up=[]
dsc = np.zeros((num_test,1))
recall = np.zeros_like(dsc)
tn = np.zeros_like(dsc)
prec = np.zeros_like(dsc)
iou = np.zeros_like(dsc)
spec = np.zeros_like(dsc)
#define F1
f1_score = np.zeros_like(dsc)

thresh = 0.5

# check the predictions from the trained model 
for i in range(num_test):
    #gt = orig_masks[testIdx[i]]
    name = img_list[testIdx[i]]
    gt = plt.imread(os.path.join(orig_dir, name.split('.')[0] + "_segmentation.png")) 

    pred_up = cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    dsc[i] = utils.check_preds(pred_up > thresh, gt)
    recall[i], spec[i], prec[i], iou[i] = utils.auc(gt, pred_up >thresh)
    
avg_dsc = np.sum(dsc)/num_test
avg_recall = np.sum(recall)/num_test
avg_precision = np.sum(prec)/num_test
f1_score = 2*avg_recall*avg_precision/(avg_precision+avg_recall)
avg_iou = np.sum(iou)/num_test
avg_specificity = np.sum(spec)/num_test
print('-'*30)
print('At threshold =', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision \t{2:^.3f} \n Specificity \t{3:^.3f} \n IOU \t\t{4:^.3f} \n F1 \t\t{5:^.3f}'.format(
        avg_dsc,  
        avg_recall,
        avg_precision,
        avg_specificity,
        avg_iou,
        f1_score))

# check the predictions with the best saved model from checkpoint
model.load_weights("weights.hdf5")
_,_,_,preds = model.predict(imgs_test)
#preds = model.predict(imgs_test)   #use this if the model is unet

preds_up=[]
dsc = np.zeros((num_test,1))
recall = np.zeros_like(dsc)
tn = np.zeros_like(dsc)
prec = np.zeros_like(dsc)
iou = np.zeros_like(dsc)
#define F1
f1_score = np.zeros_like(dsc)
spec = np.zeros_like(dsc)
for i in range(num_test):
    #gt = orig_masks[testIdx[i]]
    name = img_list[testIdx[i]]
    gt = plt.imread(os.path.join(orig_dir, name.split('.')[0] + "_segmentation.png")) 

    pred_up = cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    dsc[i] = utils.check_preds(pred_up > thresh, gt)
    recall[i], spec[i], prec[i], iou[i] = utils.auc(gt, pred_up >thresh)
    
avg_dsc = np.sum(dsc)/num_test
avg_recall = np.sum(recall)/num_test
avg_precision = np.sum(prec)/num_test
f1_score = 2*avg_recall*avg_precision/(avg_precision+avg_recall)
avg_iou = np.sum(iou)/num_test
avg_specificity = np.sum(spec)/num_test
print('-'*30)
print('USING HDF5 saved model at thresh=', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision \t{2:^.3f} \n Specificity \t{3:^.3f} \n IOU \t\t{4:^.3f} \n F1 \t\t{5:^.3f}'.format(
        avg_dsc,  
        avg_recall,
        avg_precision,
        avg_specificity,
        avg_iou,
        f1_score))
        
end_whole = datetime.datetime.now()

print("whole run time:", (end_whole - start_whole))