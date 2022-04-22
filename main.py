import yaml
import argparse
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import models.dnn, models.rnn, models.random_regression_forest
from clean_data import selecting_most_populated_columns, drop_rows_with_half_missing_values, drop_rows_where_SALEQ_ATQ_missing, imputation_softimpute, drop_rows_where_SALEQ_ATQ_zero, exclude_quarters_with_no_accouncement_date, normalize_data

parser = argparse.ArgumentParser(description='Machine Learning-Based Financial Statement Analysis')
parser.add_argument('--config', default='./configs/config.yaml')

class QuarterlyFundamentalData(Dataset):
    def __init__(self, filename):
        dataset = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.x = torch.from_numpy(dataset[:, 1:]) # Skip the column that is the target
        self.y = torch.from_numpy(dataset[:, [0]]) # Size = (n_samples, 1)
        self.num_samples = dataset.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc

def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model.forward(data)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))

def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 1
    cm = torch.zeros(num_class, num_class)
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        if torch.cuda.is_avaliable():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            out = model.forward(data)
            loss = criterion(out, target)

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Get args
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    #Set batch_size and epochs
    num_epochs = args.epoch
    batch_size = args.batch_size

    #Load data
    #dataset = pd.read_csv('data/cleaned_data.csv')
    train_set_filename = 'data/cleaned_data.csv'
    val_set_filename = ''
    train_dataset = QuarterlyFundamentalData(train_set_filename)
    data_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=False, num_workers=2) # num_workers uses multiple subprocesses

    val_dataset = QuarterlyFundamentalData(val_set_filename)
    val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False, num_workers=2) # num_workers uses multiple subprocesses


    #Set up model
    
    if args.model == 'DNN':
        model = models.dnn.DNN().to(device)
        print("DNN")
    elif args.model == 'RNN':
        model = models.rnn.RNN(device).to(device)
        print("RNN")
    elif args.model == 'RandomForest':
        model = models.random_regression_forest.RandomForest().to(device)
        print("RandomForest")
    
    #Initialize loss and optimizer
    if args.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss_type == 'MAE':
        criterion = nn.L1Loss()

    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 
    elif args.optimizer_type == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=args.learning_rate) 

    best = 0.0
    best_cm = None
    best_model = None

    for epoch in range(num_epochs):
        #Train the model
        train(epoch, data_loader, model, optimizer, criterion)

        #Validate the model
        acc, cm = validate(epoch, val_loader, model, criterion)



if __name__ == "__main__":
    main()