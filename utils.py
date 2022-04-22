import numpy as np
from timeit import default_timer as timer
import matplotlib.pylab as plt 
import pandas as pd

def split_train_test_data():
    dataset = pd.read_csv('data/cleaned_data.csv')
    separator_factor = int(0.8 * len(dataset))
    tic = dataset.iloc[separator_factor]['tic']
    #print(separator_factor, tic)
    while dataset.iloc[separator_factor + 1]['tic'] == tic:
        separator_factor += 1

    #print(separator_factor, tic)
    train_data = dataset[:separator_factor+1]
    val_data = dataset[separator_factor+1:]

    train_data.to_csv('data/train_data.csv', index=False)
    val_data.to_csv('data/val_data.csv', index=False)
    

split_train_test_data()