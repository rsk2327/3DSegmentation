import pandas as pd
import numpy as np
import os

from tqdm import tqdm

import nibabel as nib

import matplotlib.pyplot as plt

from PIL import Image


import sys
import torch
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
# from models import *

from torchvision.transforms import Resize,ToTensor, RandomHorizontalFlip, RandomVerticalFlip,Normalize
import PIL
import timm
import imgaug as ia
from imgaug import augmenters as iaa

sys.path.insert(0,'/home/roshansk/Covid/CXRData/')
from SegLearner import *

import segmentation_models_pytorch as smp

sys.path.insert(0,'/home/roshansk/Covid/RibFrac/Models/')
from models import *


trainFolder = '/home/roshansk/Covid/3dSeg/Liver/Data/OrigData/'
sliceFolder = '/home/roshansk/Covid/3dSeg/Liver/Data/Slices/'

dataFiles = os.listdir(trainFolder)

imgFiles = [x for x in dataFiles if 'orig' in x]

sliceList = sorted(os.listdir(sliceFolder))
imgFiles = [x for x in sliceList if 'image' in x]
labelFiles = [x.replace('image','label') for x in imgFiles]

df = pd.DataFrame({'imgPath':imgFiles,'maskPath':labelFiles})
df.imgPath = df.imgPath.apply(lambda x : os.path.join(sliceFolder, x))
df.maskPath = df.maskPath.apply(lambda x : os.path.join(sliceFolder, x))

def getAxis(x):
    
    if 'x' in x:
        return 'x'
    elif 'z' in x:
        return 'z'
    elif 'y' in x:
        return 'y'
    

df['axis'] = df.imgPath.apply(lambda x : getAxis(x))

df['patient'] = df.imgPath.apply(lambda x : x.split("/")[-1].split("_")[1])

print(len(df))

class SegDataset(torch.utils.data.Dataset):
    
  def __init__(self, df, imgTransforms = None, maskTransforms = None, preload = False, imgSize = 256):


    self.df = df
    self.transforms = transforms
    self.preload = preload
    self.imgSize = imgSize
    self.imgTransforms = imgTransforms
    self.maskTransforms = maskTransforms
    
    if self.preload:
        self.preloadData()
    
    
  def preloadData(self):
    imgList = []
    maskList = []
    
    for i in tqdm(range(len(self.df))):
            
            img = PIL.Image.open(self.df.iloc[i]['imgPath'])

            img = torchvision.transforms.Resize( (self.imgSize, self.imgSize))(img)
            imgList.append(np.array(img))

            
            mask = PIL.Image.open(self.df.iloc[i]['maskPath'])

            mask = torchvision.transforms.Resize( (self.imgSize, self.imgSize))(mask)
            maskList.append(np.array(mask))

            
    self.imgData = np.array(imgList)
    self.maskData = np.array(maskList)
    del imgList, maskList


  def __len__(self):

    return len(self.df)

  def __getitem__(self, index):
        
    if self.preload:
        img = self.imgData[index,:,:,:]
        img = PIL.Image.fromarray(img)
        
        mask = self.maskData[index,:,:,:]
        mask = PIL.Image.fromarray(mask)
        
        
    else:
        imgFilename = self.df.iloc[index]['imgPath']
        maskFilename = self.df.iloc[index]['maskPath'] 
        
        img = Image.open(imgFilename).convert('RGB')
        mask = Image.open(maskFilename)
        
    
    
    
    if self.imgTransforms:
        img = self.imgTransforms(img)
    
    if self.maskTransforms:
        mask = self.maskTransforms(mask)
        
    # If mask has multiple channels
    if mask.shape[0]!=1:
        mask = mask[0,:,:]
        
    mask[mask>0] = 1
    
#     mask = mask.type(torch.uint8)

    return img, mask



class ImgAugTransform:
  def __init__(self ):
        
    self.aug = iaa.Sequential([
#         iaa.HorizontalFlip(p = 0.5),
#         iaa.VerticalFlip(p = 0.5),
#         iaa.Affine(scale=(0.5, 1.5)),
        iaa.Dropout(p=(0, 0.2), per_channel=0.5),
        iaa.SomeOf((1,2),[
#                     iaa.Cutout(fill_mode="gaussian", fill_per_channel=True),
                    iaa.SaltAndPepper(0.1),
#                     iaa.Affine(rotate=(-45, 45), shear=(-16, 16)),
#                     iaa.imgcorruptlike.GaussianNoise(severity=1),
#                     iaa.AveragePooling(2),
                    iaa.AddToHueAndSaturation((-60, 60)),
                    iaa.MultiplyBrightness(mul=(0.65, 1.35)),
                    iaa.LinearContrast((0.5, 2.0)),
                    iaa.GaussianBlur(sigma=(0.5, 2.0)),
#                     iaa.CoarseDropout((0.01,0.1), size_percent = 0.01)
        ])
    ])
    
    
      
  def __call__(self, img):
    img = np.array(img)
    return PIL.Image.fromarray(self.aug.augment_image(img))
 
    
    





device = 'cuda:1'

batchSize = 10
imgSize = 256


# model = ResNetDUC(num_classes=1)
model = smp.Unet('resnet50', encoder_weights='imagenet', classes=1, activation='sigmoid')
# model = smp.Unet('resnet34', encoder_weights='imagenet')
# model = smp.PSPNet('resnet34', encoder_weights='imagenet')



criterion = torch.nn.BCELoss()


optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

model.to(device)


trainTransforms = transforms.Compose([transforms.Resize((imgSize,imgSize)), 
                                      ToTensor()])


imgTrainTransforms = transforms.Compose([Resize( (imgSize, imgSize) ),
                                      ImgAugTransform(),
                                       ToTensor(),
                                       Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])

maskTrainTransforms = transforms.Compose([Resize( (imgSize, imgSize) ),
                                       ToTensor()])


imgTestTransforms = transforms.Compose([Resize( (imgSize, imgSize) ),
                                       ToTensor(),
                                       Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])


# trainTransforms = transforms.Compose([transforms.RandomResizedCrop(size = 256,scale = (0.06,0.5), ratio=(0.75, 1.3)), 
#                                       ToTensor()])

classLoss = nn.CrossEntropyLoss()
reconLoss = nn.MSELoss()

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


trainDataset = SegDataset(df.iloc[0:6270], imgTrainTransforms, maskTrainTransforms, preload=False, imgSize = imgSize)
testDataset = SegDataset(df.iloc[6270:len(df)], imgTestTransforms, maskTrainTransforms, preload =False, imgSize = imgSize)

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, 
                                          shuffle=True, num_workers=6)


testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, 
                                          shuffle=False, num_workers=6)



learner = SegLearner(model, trainLoader, optimizer, criterion, testLoader, device = device)


lr = 0.001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.0001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.00001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=20, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.000001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.000001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=20, save_best_model=False, save_every_epoch=False, useLogger = False)

