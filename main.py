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
from dcnn import DCNN
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
        batch_size = target.size(0)
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)
        total += len(target)
        return 100.0 * float(correct) / float(total)

def print_matrix(output, target):
    with torch.no_grad():
         pre, tar = [], []
         _, predicted = torch.max(output.data, 1)
         tar += target.view(-1).tolist()
         pre += predicted.view(-1).tolist()
         con_x = metrics.confusion_matrix(tar, pre)
         print(con_x)



for i in range(1):
    numnum = str( i + 2 )
    
    BATCH_SIZE = 64

    all_train_file_name,all_train_file_label,all_test_file_name,all_test_file_label = dataloader_diao()

    TrainImgLoader = torch.utils.data.DataLoader(
             myImageFloder(all_train_file_name,all_train_file_label, True), 
             batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=True)

    testImgLoader = torch.utils.data.DataLoader(
             myImageFloder(all_test_file_name,all_test_file_label, False), 
             batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=False)

    #resnet = models.inception_v3(pretrained=True).cuda()
    #resnet.aux_logits = False
    #resnet = MSnet(adaptive=False)


    inception_v3 = models.inception_v3(pretrained=True)
    inception_v3.AuxLogits.fc = nn.Linear(inception_v3.AuxLogits.fc.in_features, 3)
    inception_v3.fc = nn.Linear(2048, 3)

    #densenet121 = models.densenet121(pretrained=True)
    #densenet121.classifier = nn.Linear(densenet121.classifier.in_features, 3)
    resnet = models.inception_v3(pretrained=False, num_classes=3)
    resnet.aux_logits = False
    pre_dict = inception_v3.state_dict()
    model_dict = resnet.state_dict()

    pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    model_dict.update(pre_dict)
    resnet.load_state_dict(model_dict)


    
    resnet.cuda()
    # Loss and Optimizer
    weights = torch.tensor([1.0, 2.5, 3.9]).cuda()   #classes = 3  
    #ori: [1.0, 2.999, 2.749]     [1.0, 2.5, 4.2]-120921:37
    #[1.0, 2.5, 5.2/4.0/3.9]-120921:47
    #weights = torch.tensor([1.0, 1.5]).cuda()      #classes = 2
    criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    lr = 0.001
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)
    PATH = '/home/dsh/hello/model/NPC/box/pre_inception_v3_%s.pth'%numnum


    # Training
    EPOCH = 30
    losses = 1000
    for epoch in range(EPOCH):
        for i, (images, labels) in enumerate(TrainImgLoader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            resnet.train()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % (int(len(TrainImgLoader)/4)) == 0:
                print("Epoch [%d/%d], Iter [%d/%d],  Loss= %.4f, ACC= %.2f %%" % (epoch + 1, EPOCH, i + 1, len(TrainImgLoader), loss.item(), acc))
        with torch.no_grad():
            t0 = time.time()
            resnet.eval()   
            loss = 0
            acc1 = 0
            for i, (images, labels) in enumerate(testImgLoader):
                test_image = Variable(images.cuda())
                the_labels = Variable(labels.cuda())
                outputs = resnet(test_image)
                loss += criterion(outputs, the_labels)
                acc1 += accuracy(outputs, the_labels)
            loss /= i + 1
            t1 = time.time()
            print('val_loss= %.4f, ACC= %.2f %%'%(loss.item(), acc1 / (i + 1)))
            print('Mean EPOCH spent time is : %.4f sec.' % ((t1 - t0)/i+1))
            #print_matrix(outputs, the_labels)
            if loss < losses:
                losses = loss
                torch.save(resnet.state_dict(), PATH)
        
            # Decaying Learning Rate
        scheduler.step()


    resnet.load_state_dict(torch.load(PATH))
    resnet.eval().cuda()
    with torch.no_grad():
         pre, tar = [], []
         for i, (images, labels) in enumerate(testImgLoader):
             images = Variable(images.cuda())
             labels = Variable(labels.cuda())
             optimizer.zero_grad()
             outputs = resnet(images)
             _, predicted = torch.max(outputs.data, 1)
             tar += labels.view(-1).tolist()
             pre += predicted.view(-1).tolist()
         con_x = metrics.confusion_matrix(tar, pre) 
         print('The %s result of train'%numnum)
         print(con_x)



