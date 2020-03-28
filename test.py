from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
import math
from data_loader import myImageFloder
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from dcnn import *
from mul_scale import *

torch.manual_seed(1314)
torch.cuda.manual_seed(1314)


def dataloader_diao():

    #get the test data and test label

    return all_test_file_name,all_test_file_label



def accuracy(output, target):
    correct = 0
    total = 0
    with torch.no_grad():
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)
        total += len(target)
        return 100.0 * float(correct) / float(total)




BATCH_SIZE = 128

all_test_file_name,all_test_file_label = dataloader_diao()


testImgLoader = torch.utils.data.DataLoader(
         myImageFloder(all_test_file_name,all_test_file_label, False), 
         batch_size= BATCH_SIZE, shuffle= True, num_workers=4, drop_last=False)

resnet = models.inception_v3(pretrained=False, num_classes=3)
resnet.aux_logits = False
resnet = resnet.cuda()

losses = nn.CrossEntropyLoss()
PATH = '....pth'



correct, total = 0, 0
resnet.load_state_dict(torch.load(PATH))
resnet.eval().cuda()
with torch.no_grad():
    pre, tar = [], []
    for i, (images, labels) in enumerate(TrainImgLoader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        tar += labels.view(-1).tolist()
        pre += predicted.view(-1).tolist()
        correct += torch.sum(predicted == labels)
        con_x = metrics.confusion_matrix(tar, pre)
    print(con_x)
    print('tar is :', tar)
    print('pre is :', pre)
    

