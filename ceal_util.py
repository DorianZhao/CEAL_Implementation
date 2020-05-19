#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import Optional, Callable
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import torch.optim as optim
import torch
import torch.optim as Optimizer
import logging


# In[30]:


#Simple lenet model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


# In[36]:


#Encapsulate LeNet
class LeNet(object):
    def __init__(self, n_classes = 10, device = None):
        self.n_classes = n_classes
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        
    def __train_one_epoch__(self, train_loader, optimizer, criterion,
                        valid_loader = None, epoch = 0, each_batch_idx = 0):
        train_loss = 0
        data_size = 0
        
        for batch_idx, (img, label) in enumerate(train_loader):
            
            img = img.to(self.device)
            label = label.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # run forward
            pred_prob = self.model(img)

            # calculate loss
            loss = criterion(pred_prob, label)

            # calculate gradient (backprop)
            loss.backward()

            # total train loss
            train_loss += loss.item()
            data_size += label.size(0)

            # update weights
            optimizer.step()
            
        if valid_loader:
            acc = self.evaluate(test_loader=valid_loader)
            print('Accuracy on the valid dataset {}'.format(acc))
            
    def train(self, epochs, train_loader, valid_loader = None):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD( self.model.parameters()),
            lr=0.8)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.__train_one_epoch__(train_loader=train_loader,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   valid_loader=valid_loader,
                                   epoch=epoch
                                   )

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data,labels) in enumerate(test_loader):
                data = data.float()
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def predict(self, test_loader):
        self.model.eval()
        self.model.to(self.device)
        predict_results = np.empty(shape=(0, 10))
        with torch.no_grad():
            for batch_idx,  (img, label) in enumerate(test_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                outputs = self.model(img)
                outputs = softmax(outputs)
                predict_results = np.concatenate(
                    (predict_results, outputs.cpu().numpy()))
        return predict_results





