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


sys.path.insert(0,'/home/roshansk/Covid/CXRData/')
from SegLearner import *

import segmentation_models_pytorch as smp

sys.path.insert(0,'/home/roshansk/Covid/RibFrac/Models/')
from models import *


trainFolder = '/home/roshansk/Covid/Live/ircad-dataset/train/'
sliceFolder = '/home/roshansk/Covid/Live/ircad-dataset/Slices/'

dataFiles = os.listdir(trainFolder)

imgFiles = [x for x in dataFiles if 'orig' in x]

sliceList = sorted(os.listdir(sliceFolder))
imgFiles = [x for x in sliceList if 'image' in x]
labelFiles = [x.replace('image','label') for x in imgFiles]

df = pd.DataFrame({'imgPath':imgFiles,'maskPath':labelFiles})
df.imgPath = df.imgPath.apply(lambda x : os.path.join(sliceFolder, x))
df.maskPath = df.maskPath.apply(lambda x : os.path.join(sliceFolder, x))

print(len(df))




device = 'cuda:1'

batchSize = 10
imgSize = 256


model = ResNetDUC(num_classes=1)

# model = smp.Unet('resnet34', encoder_weights='imagenet')
# model = smp.PSPNet('resnet34', encoder_weights='imagenet')



criterion = torch.nn.BCELoss()


optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

model.to(device)


trainTransforms = transforms.Compose([transforms.Resize((imgSize,imgSize)), 
                                      ToTensor()])


# trainTransforms = transforms.Compose([transforms.RandomResizedCrop(size = 256,scale = (0.06,0.5), ratio=(0.75, 1.3)), 
#                                       ToTensor()])

classLoss = nn.CrossEntropyLoss()
reconLoss = nn.MSELoss()

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


trainDataset = SegDataset(df.iloc[0:1650], trainTransforms)
testDataset = SegDataset(df.iloc[1650:len(df)], trainTransforms)

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, 
                                          shuffle=True, num_workers=8)


testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, 
                                          shuffle=False, num_workers=8)



learner = SegLearner(model, trainLoader, optimizer, criterion, testLoader, device = device)


lr = 0.001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.0001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.00001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)

lr = 0.000001
learner.optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
learner.fit(num_epochs=4, save_best_model=False, save_every_epoch=False, useLogger = False)