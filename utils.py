import numpy as np
from numpy import genfromtxt
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

def generate_batches(filename, model):
    dataset = pd.read_csv(filename)
    new_dataset = pd.DataFrame()
    if model == 'RNN':

        dataset = dataset.drop(['tic', 'datadate', 'PRC'])
    else:
        tickers = dataset['tic'].unique()
        for tic in tickers:
            tmp_input_arr = np.array([])
            cond = dataset['tic'] == tic
            ticker_stock_data = dataset.loc[cond]
            ticker_stock_data.reset_index(drop=True, inplace=True)
            ticker_stock_data = ticker_stock_data.drop(['tic', 'datadate', 'PRC'], axis=1)
            num_rows = len(ticker_stock_data)
            for row in range(4, num_rows):
                tmp = pd.concat([ticker_stock_data.loc[row - 4, ticker_stock_data.columns != 'BHAR'], ticker_stock_data.loc[row - 3, ticker_stock_data.columns != 'BHAR'], ticker_stock_data.loc[row - 2, ticker_stock_data.columns != 'BHAR'], ticker_stock_data.loc[row - 1, ticker_stock_data.columns != 'BHAR'], pd.DataFrame([ticker_stock_data.loc[row]['BHAR']])], axis=0, join='outer', ignore_index=True).T
                new_dataset = pd.concat([new_dataset, tmp], ignore_index=True)
                #new_dataset = new_dataset.append(tmp, ignore_index=True)
                
                
        new_dataset.to_csv('data/batched_train_data.csv', index=False, header=False)
        print(f'Number of rows: {new_dataset.shape[0]}, Number of columns: {new_dataset.shape[1]}')
            
def change_data_to_float(filename):
    #data = pd.read_csv(filename, nrows=20).to_numpy()
    data = genfromtxt(filename, delimiter=',')
    data = pd.DataFrame(data.astype(np.float32))
    data.to_csv('data/batched_val_data.csv', index=False, header=False)
        

#split_train_test_data()
#generate_batches('data/train_data.csv', "DNN")
#change_data_to_float('data/batched_val_data.csv')
