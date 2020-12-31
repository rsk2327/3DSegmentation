import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import nibabel as nib

from niwidgets import NiftiWidget

import marshal

import niwidgets
import matplotlib.pyplot as plt


from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
sys.path.insert(0,'/home/roshansk/Covid/RibFrac/Models')
from models import *

import time
import itertools
import logging
import datetime

import copy



class SegLearner():
    
    def __init__(self, model, trainLoader, optimizer, criterion, validLoader, device='cuda:0',
                logger = None, modelFolder = "./", modelname_prefix = "", progressSaveFolder = '',
                saveProgressImages = False):
        
        
        self.model = model
        
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.trainLoader = trainLoader
        self.validLoader = validLoader
        
        self.progressSaveFolder = progressSaveFolder
        self.saveProgressImages = saveProgressImages
        
        self.modelFolder = modelFolder
        self.modelname_prefix = modelname_prefix
        
        self.device = device
        self.model = self.model.to(self.device)
        
        
        self.epoch = 0
        
        
        if logger is None:
            logging.basicConfig(filename='./log.txt',level=logging.DEBUG, format='%(message)s', filemode='w')
            self.logger = logging.getLogger()
        elif logger is False:
            self.logger = DummyLogger()
        else:
            self.logger = logger
            
                                
            
        
        
        
    def fit(self, num_epochs, save_best_model = True, save_every_epoch = False, useLogger = True):
        
        if save_every_epoch:
            save_best_model = False
            
        best_metric = 0.0
        best_epoch = 0
        best_model = None
        
        for epoch in range(num_epochs):
            self.epoch +=1
            
            epoch_start_time = time.time()

            self.model.train()

            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.logger.info('-' * 10)
            
            self.model.train()
            
            running_loss = 0.0
            running_corrects = 0
            
            
            for data,mask in tqdm(self.trainLoader, miniters = int(len(self.trainLoader)/50), mininterval = 30 ):
                data,mask = data.to(self.device), mask.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                loss = self.criterion(outputs.view(-1), mask.view(-1))

                loss.backward()
                self.optimizer.step()
                  
                running_loss += loss.item()
                
                
                
            epoch_loss = running_loss/len(self.trainLoader)
            
            epochIOU = np.mean(self.evalModel(self.trainLoader))
            
            if useLogger:
                self.logger.info(f"Epoch Time : {str(datetime.timedelta(seconds = time.time() - epoch_start_time))}")
                self.logger.info(f"Train Loss : {epoch_loss}  Train IOU : {epochIOU}")
            else:
                print(f"Epoch Time : {str(datetime.timedelta(seconds = time.time() - epoch_start_time))}")
                print(f"Train Loss : {epoch_loss}  Train IOU : {epochIOU}")
            
            
            val_epoch_iou = np.mean(self.evalModel(self.validLoader))
            
            if useLogger:
                self.logger.info(f"Test IOU : {val_epoch_iou}")
            else:
                print(f"Test IOU : {val_epoch_iou}")
                
                
            if val_epoch_iou > best_metric:
                best_metric = val_epoch_iou
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = self.epoch + epoch

            if save_every_epoch:
                self.saveModel(epoch + self.epoch, val_epoch_iou)
                
                
            
                
        if save_best_model:
            self.model.load_state_dict(best_model)
            self.saveModel(best_epoch, best_metric)
            
            
    def saveModel(self, epoch, iou):
        
        iou = np.round(iou*100)

        filename = f"{self.modelname_prefix}_{epoch}_{iou}.pt"

        path = os.path.join(self.modelFolder, filename)
        torch.save(self.model, path)
        
        
        
    def predict(testLoader):
        
        self.model.eval()
        
        targetList = []
        predList = []
        
        with torch.no_grad():
            
            for data, mask in tqdm(testLoader):

                data,mask = data.to(self.device), mask.to(self.device)
                outputs = self.model(data)


                targetList.append(labels.detach().cpu().numpy())
                predList.append(preds.detach().cpu().numpy())

        

        return targetList, predList
            
            
            
    def evalModel(self, testLoader, threshold = 0.5):
        
        """
        Evaluation of the model is done using IOU metric
        """
    
        self.model.eval()
        
        iouScores = []

        for (img, mask) in tqdm(testLoader):

            img, mask = img.to(self.device), mask.to(self.device)

            out = self.model(img)
            out = out>threshold

            iou = list(self.iou_pytorch(out.long(), mask.long().squeeze(1)).detach().cpu().numpy())
            iouScores.append(iou)

        self.model.train()

        return list(itertools.chain.from_iterable(iouScores))
    
    
    
    def checkProgress(self):
    
        self.model.eval()
        
        img, mask = next(iter(self.validLoader))
        
        out = self.model(img.to(self.device)).detach().cpu().numpy()
        
        
        numFiles = img.shape[0]

        plt.figure(figsize = (10,5))

        for i in range(numFiles):
        
            plt.subplot(2,numFiles, i+1)
            plt.imshow(mask[i,0,:,:])



            plt.subplot(2, numFiles,numFiles + i+1 )
            output = out[i,0,:,:] > 0.5
            plt.imshow(output)


        self.model.train()

    

    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        SMOOTH = 1e-6

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded  # Or thresholded.mean() if you are interested in average across the batch


                
                
            

            
class SegDataset(torch.utils.data.Dataset):
    
  def __init__(self, df, transforms = None):


    self.df = df
    self.transforms = transforms


  def __len__(self):

    return len(self.df)

  def __getitem__(self, index):
        
        
    imgFilename = self.df.iloc[index]['imgPath']
    maskFilename = self.df.iloc[index]['maskPath']    
        
    
    img = Image.open(imgFilename).convert('RGB')
    mask = Image.open(maskFilename)
    
    if self.transforms:
        
        img = self.transforms(img)
        mask = self.transforms(mask)
        
    # If mask has multiple channels
    if mask.shape[0]!=1:
        mask = mask[0,:,:]
        
    mask[mask>0] = 1
    
#     mask = mask.type(torch.uint8)

    return img, mask
            




class DummyLogger():
    
    def __init__(self):
        a =1
        
    def info(self,x):
        print(x)