'''
Author: Peng Bo
Date: 2022-08-11 21:05:24
LastEditTime: 2022-08-27 18:22:00
Description: 

'''
# coding: utf8
from __future__ import print_function

import argparse
import yaml
import os
import random
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from models import MLNet
import dataset


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()

    sample_sum  = 0
    correct_sum = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        sample_sum += output.size(0)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_sum += pred.eq(target.view_as(pred)).sum().item()
        
        
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct_sum/sample_sum))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(config_file):
    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    commconfig = config['common']
    commconfig['save_path'] = os.path.dirname(config_file)
    
    # prepare dataset and dataloader
    datconfig = config['dataset']
    # print('==> Preparing dataset %s' % datconfig['type'])
    # create dataset for training and testing
    trainset = dataset.__dict__[datconfig['type']](datconfig['train_list'], datconfig['train_meta'])
    testset = dataset.__dict__[datconfig['type']](datconfig['test_list'], datconfig['test_meta'])
    # create dataloader for training and testing
    trainloader = data.DataLoader(
        trainset, batch_size=commconfig['train_batch'], shuffle=True, num_workers=1)
    testloader = data.DataLoader(
        testset, batch_size=commconfig['test_batch'], shuffle=False, num_workers=1)

    torch.manual_seed(random.randint(0, 1000))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare model, optimizer, scheduler
    model = MLNet(input_dim=trainset.feature_dim(), num_classes=trainset.feature_dim()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=commconfig['lr'])
    # scheduler = StepLR(optimizer, step_size=1, gamma=commconfig['gamma'])
    
    # start training
    for epoch in range(1, commconfig['epoch'] + 1):
        train(args, model, device, trainloader, criterion, optimizer, epoch)
        test(model, device, criterion, testloader)

    torch.save(model.state_dict(), os.path.join(commconfig['save_path'], 'weight.pt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pose state classify')
    parser.add_argument('--config-file', type=str,default='experiments/template/config.yaml')
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    main(args.config_file)