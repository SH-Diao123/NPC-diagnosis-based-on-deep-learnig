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

    a_path = '/home/dsh/Data/NPC/train/0/'
    b_path = '/home/dsh/Data/NPC/train/1/'
    c_path = '/home/dsh/Data/NPC/train/2/'
    a_file = os.listdir(a_path)
    b_file = os.listdir(b_path)
    c_file = os.listdir(c_path)

    all_train_file_name, all_train_file_label = [], []
    all_test_file_name, all_test_file_label = [], []

    for i in range(0, len(a_file)):
        all_train_file_name.append(a_path+a_file[i])
        all_train_file_label.append(0)
    for i in range(0, len(b_file)):
        all_train_file_name.append(b_path+b_file[i])
        all_train_file_label.append(1)
    for i in range(0, len(c_file)):
        all_train_file_name.append(c_path+c_file[i])
        all_train_file_label.append(2)


    aa_path = '/home/dsh/Data/NPC/test/0/'
    bb_path = '/home/dsh/Data/NPC/test/1/'
    cc_path = '/home/dsh/Data/NPC/test/2/'
    aa_file = os.listdir(aa_path)
    bb_file = os.listdir(bb_path)
    cc_file = os.listdir(cc_path)


    for i in range(0, len(aa_file)//4*3):
        all_test_file_name.append(aa_path+aa_file[i])
        all_test_file_label.append(0)
    for i in range(0, len(bb_file)//4*3):
        all_test_file_name.append(bb_path+bb_file[i])
        all_test_file_label.append(1)
    for i in range(0, len(cc_file)//4*3):
        all_test_file_name.append(cc_path+cc_file[i])
        all_test_file_label.append(2)

    return all_train_file_name,all_train_file_label,all_test_file_name,all_test_file_label



def accuracy(output, target):
    correct = 0
    total = 0
    with torch.no_grad():
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)
        total += len(target)
        return 100.0 * float(correct) / float(total)




BATCH_SIZE = 512

all_train_file_name,all_train_file_label,all_test_file_name,all_test_file_label = dataloader_diao()

TrainImgLoader = torch.utils.data.DataLoader(
             myImageFloder(all_train_file_name,all_train_file_label, True), 
             batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=True)
testImgLoader = torch.utils.data.DataLoader(
         myImageFloder(all_test_file_name,all_test_file_label, False), 
         batch_size= BATCH_SIZE, shuffle= True, num_workers=4, drop_last=False)

#resnet = MSnet(adaptive=True)
resnet = models.inception_v3(pretrained=False, num_classes=3)
resnet.aux_logits = False
resnet = resnet.cuda()

losses = nn.CrossEntropyLoss()
PATH = '/home/dsh/hello/model/NPC/box/pre_inception_v3_5.pth'



correct, total = 0, 0
TN, TP, FN, FP = 0, 0, 0, 0
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
    
    
     #compute auc and show
    pre = label_binarize(pre, classes=[0, 1, 2])
    tar = label_binarize(tar, classes=[0, 1, 2])
    fpr, tpr, auc_value = dict(), dict(), dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(tar[:, i], pre[:, i])
        auc_value[i] = auc(fpr[i], tpr[i])
    print('auc_value is', auc_value)
    print('fpr is ', fpr)
    print('tpr is ', tpr)
    
    plt.figure()
    plt.plot(fpr[0], tpr[0], color='green', lw=2, label='Inflammation(AUC={:2f})'.format(auc_value[0]))
    plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Lymph(AUC={:2f})'.format(auc_value[1]))
    plt.plot(fpr[2], tpr[2], color='blue', lw=2, label='Cancer(AUC={:2f})'.format(auc_value[2]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
    

'''
def bingren():
    all_test_file_name,all_test_file_label = big_diao()
    
    new_file, tmp_file = [], []
    for i in all_test_file_name:
        path = os.path.dirname(i) + '/'
        name = i.split('/')[-1]
        c_name = name.split(' ')[0] + ' ' + name.split(' ')[1] + ' '
        for j in range(1, 10):
            if os.path.exists(path+c_name+str(j)+'.jpg'):
               tmp_file.append(path+c_name+str(j)+'.jpg')
            else:
               all_file.append(tmp_file)
               tmp_file = []
               break
    testImgLoader = torch.utils.data.DataLoader(
         myImageFloder(all_test_file_name,all_test_file_label, True), 
         batch_size= 1, shuffle= False, num_workers= 1)

    correct, total = 0, 0
    TN, TP, FN, FP = 0, 0, 0, 0
    resnet.load_state_dict(torch.load(PATH))
    resnet.eval().cuda()
    with torch.no_grad():
         pre, tar = [], []
    for i, (image, label) in enumerate(testImgLoader):
'''   
