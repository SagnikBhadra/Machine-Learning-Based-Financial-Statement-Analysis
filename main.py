import yaml
import argparse
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt 

import models.dnn, models.rnn, models.random_regression_forest, models.LASSO, models.OLS
from utils import calculate_error_per_epsilon, plot_losses, percentage_correct

parser = argparse.ArgumentParser(description='Machine Learning-Based Financial Statement Analysis')
parser.add_argument('--config', default='./configs/config.yaml')

class QuarterlyFundamentalData(Dataset):
    def __init__(self, filename):
        dataset = np.loadtxt(filename, delimiter=",")
        self.x = torch.from_numpy(dataset[:, :484]) # Skip the column that is the target
        self.y = torch.from_numpy(dataset[:, [484]]) # Size = (n_samples, 1)
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

def ML_train(epoch, data_loader, model):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        data = data.cpu().float().numpy()
        target = target.cpu().float().numpy()
        model.train(data, target)
        iter_time.update(time.time() - start)


def ML_validation(epoch, val_loader, model, criterion, percentage_correct_criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    all_losses = []

    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        data = data.cpu().float().numpy()
        target = target.float()

        out = torch.tensor(model.test(data))#.unsqueeze(1)
        loss = criterion(out, target)

        all_losses.append(percentage_correct_criterion(out, target))
        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))

    return all_losses, losses.avg.tolist()
    

def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        data = data.float()
        target = target.float()
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
        
    return losses.avg.tolist()

def validate(epoch, val_loader, model, criterion, percentage_correct_criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    all_losses = []

    num_class = 1
    cm = torch.zeros(num_class, num_class)
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        data = data.float()
        target = target.float()

        with torch.no_grad():
            out = model.forward(data)
            loss = criterion(out, target)
        all_losses.append(percentage_correct_criterion(out, target))
        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    
    return all_losses, losses.avg.tolist()

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

    #Set batch_size and epochs and filenames
    num_epochs = args.epoch
    batch_size = args.batch_size
    train_set_filename = 'data/batched_train_data.csv'
    val_set_filename = 'data/batched_val_data.csv'
    train_losses = []
    val_losses = []
    val_all_losses = []
    val_percentage_correct = []

    val_dataset = np.loadtxt(val_set_filename, delimiter=",")
    val_target = torch.from_numpy(val_dataset[:, [484]]) 

    #Set up model
    
    if args.model == 'DNN':
        model = models.dnn.DNN().to(device)
        print("DNN")
    elif args.model == 'RNN':
        model = models.rnn.RNN(device).to(device)
        train_set_filename = 'data/batched_rnn_train_data.csv'
        val_set_filename = 'data/batch_rnn_val_data.csv'
        print("RNN")
    elif args.model == 'RandomForest':
        model = models.random_regression_forest.RandomForest().to(device)
        print("RandomForest")
    elif args.model == 'LASSO':
        model = models.LASSO.LASSO().to(device)
        print("LASSO")
    elif args.model == 'OLS':
        model = models.OLS.OLS().to(device)
        print("OLS")

    #Load data
    #dataset = pd.read_csv('data/cleaned_data.csv')
    
    train_dataset = QuarterlyFundamentalData(train_set_filename)
    data_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=False, num_workers=2) # num_workers uses multiple subprocesses

    val_dataset = QuarterlyFundamentalData(val_set_filename)
    val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False, num_workers=2) # num_workers uses multiple subprocesses

    
    #Initialize loss and optimizer
    if args.loss_type == 'MSE':
        criterion = nn.MSELoss()
        percentage_correct_criterion = nn.MSELoss(reduction='none')
    elif args.loss_type == 'MAE':
        criterion = nn.L1Loss()
        percentage_correct_criterion = nn.L1Loss(reduction='none')

    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 
    elif args.optimizer_type == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=args.learning_rate) 


    if args.model == "RNN" or args.model == "DNN":
        for epoch in range(num_epochs):
            #Train the model
            train_losses.append(train(epoch, data_loader, model, optimizer, criterion))

            #Validate the model
            val_all_losses, tmp = validate(epoch, val_loader, model, criterion, percentage_correct_criterion)
            val_losses.append(tmp)

            plot_losses(train_losses, val_losses)


    elif args.model == "OLS" or args.model == "LASSO" or args.model == "RandomForest":
        #Train the model
        ML_train(epoch, data_loader, model)
        val_all_losses, tmp = ML_validation(epoch, val_loader, model, criterion, percentage_correct_criterion)
        val_losses.append(tmp)

    BHAR = pd.DataFrame(val_target.tolist(), columns=["BHAR"])
    losses = [item for sublist in val_all_losses for item in sublist]
    val_all_losses = pd.DataFrame(losses, columns=["Loss"])

    val_all_outs = [item for sublist in val_all_outs for item in sublist]
    val_all_outs = pd.DataFrame(val_all_outs, columns=["Out"])

    calculate_error_per_epsilon(val_all_losses, BHAR)
    percentage_correct(val_all_losses, BHAR)

    if args.model == "RandomForest":
        data_columns = pd.read_csv('data/train_data.csv', nrows=1).columns
        all_columns = data_columns + data_columns + data_columns + data_columns
        plt.barh(all_columns, model.random_forest_fitted.feature_importances_)


    
            



if __name__ == "__main__":
    main()