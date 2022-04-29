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

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["Train", "Val"])
    plt.show()


def calculate_error_per_epsilon(losses, BHAR):
    data = pd.merge(losses, BHAR, how="outer", left_index=True, right_index = True)
    filt_000 = (data['BHAR'].abs() > 0) & (data['BHAR'].abs() <= 0.05)
    filt_005 = (data['BHAR'].abs() > 0.05) & (data['BHAR'].abs() <= 0.10)
    filt_010 = (data['BHAR'].abs() > 0.10) & (data['BHAR'].abs() <= 0.20)
    filt_020 = (data['BHAR'].abs() > 0.20) & (data['BHAR'].abs() <= 0.50)
    filt_050 = (data['BHAR'].abs() > 0.50) & (data['BHAR'].abs() <= 1)
    filt_100 = (data['BHAR'].abs() > 1)

    print(f'0 Error: {data.loc[filt_000].shape}\n 0.05 Error: {data.loc[filt_005].shape}\n 0.10 Error: {data.loc[filt_010].shape}\n 0.20 Error: {data.loc[filt_020].shape}\n 0.50 Error: {data.loc[filt_050].shape}\n 1.00 Error: {data.loc[filt_100].shape}\n')


    data_000 = data.loc[filt_000]["Loss"].mean()
    data_005 = data.loc[filt_005]["Loss"].mean()
    data_010 = data.loc[filt_010]["Loss"].mean()
    data_020 = data.loc[filt_020]["Loss"].mean()
    data_050 = data.loc[filt_050]["Loss"].mean()
    data_100 = data.loc[filt_100]["Loss"].mean()
    
    print(f'0 Error: {data_000}\n 0.05 Error: {data_005}\n 0.10 Error: {data_010}\n 0.20 Error: {data_020}\n 0.50 Error: {data_050}\n 1.00 Error: {data_100}\n')

def percentage_correct(outs, BHAR):
    data = pd.merge(outs, BHAR, how="outer", left_index=True, right_index = True)
    filt = ((data['BHAR'] > 0)  & (data['Out'] > 0) | (data['BHAR'] < 0)  & (data['Out'] < 0))
    filt = pd.DataFrame(filt, columns=['Percentage Correct'])
    print(f'Percentage_correct: {filt["Percentage Correct"].value_counts(normalize=True)}')
    return filt
        

#split_train_test_data()
#generate_batches('data/train_data.csv', "DNN")
#change_data_to_float('data/batched_val_data.csv')
