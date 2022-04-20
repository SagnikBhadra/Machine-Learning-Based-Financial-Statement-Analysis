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
    def __init__(self):
        dataset = np.loadtxt('data/cleaned_data.csv', delimiter=",", skiprows=1)
        self.x = torch.from_numpy(dataset[:, 1:]) # Skip the column that is the target
        self.y = torch.from_numpy(dataset[:, [0]]) # Size = (n_samples, 1)
        self.num_samples = dataset.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples

def train(epochs, data_loader, model, optimizer, criterion):
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(data_loader):
            #start = time.time()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            out = model.forward(data)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

def validate(epochs, val_loader, model, criterion):
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(val_loader):
            if torch.cuda.is_avaliable():
                data = data.cuda()
                target = target.cuda()

            with torch.no_grad():
                out = model.forward(data)
                loss = criterion(out, target)

            

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
    dataset = QuarterlyFundamentalData()
    data_loader = DataLoader(dataset=dataset, batch_size= batch_size, shuffle=True, num_workers=2) # num_workers uses multiple subprocesses

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

    #Train the model
    train(num_epochs, data_loader, model, optimizer, criterion)

    #Validate the model
    validate(num_epochs, val_loader, model, criterion)



if __name__ == "__main__":
    main()