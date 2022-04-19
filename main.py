import yaml
import argparse
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import models.dnn, models.rnn, models.random_regression_forest
from clean_data import selecting_most_populated_columns, drop_rows_with_half_missing_values, drop_rows_where_SALEQ_ATQ_missing, imputation_softimpute, drop_rows_where_SALEQ_ATQ_zero, exclude_quarters_with_no_accouncement_date, normalize_data

parser = argparse.ArgumentParser(description='Machine Learning-Based Financial Statement Analysis')
parser.add_argument('--config', default='./configs/config.yaml')

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

def validate(epochs, data_loader, model, criterion):
    pass

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




if __name__ == "__main__":
    main()