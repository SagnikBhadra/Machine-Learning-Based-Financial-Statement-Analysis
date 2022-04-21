import numpy as np
from timeit import default_timer as timer
import matplotlib.pylab as plt 
import pandas as pd
from fancyimpute import SoftImpute

def selecting_most_populated_columns(dataset):
    return dataset[['tic', 'datadate', 'acoq', 'actq', 'ancq', 'aoq', 'apq', 'atq', 'capsq', 'ceqq', 'cheq',
        'csh12q', 'cshoq', 'cshprq', 'cshpry', 'cstkq', 'dlcq', 'dlttq',
        'dpactq', 'icaptq', 'invtq', 'lcoq', 'lctq', 'lltq', 'loq', 'lseq', 'ltmibq',
        'ltq', 'mibq', 'mibtq', 'ppegtq', 'ppentq', 'pstknq', 'pstkq',
        'pstkrq', 'rectq', 'req', 'seqq', 'tstkq', 'txditcq', 'txpq', 'wcapq', 'aolochy', 'apalchy', 'aqcy', 'capxy', 'chechy', 'dltisy', 'dltry',
        'dpcy', 'dvy', 'esubcy', 'exrey', 'fiaoy', 'fincfy', 'fopoy', 'ibcy',
        'intpny', 'invchy', 'ivacoy', 'ivchy', 'ivncfy', 'ivstchy', 'oancfy',
        'prstkcy', 'recchy', 'sivy', 'sppivy', 'sstky', 'txdcy', 'xidocy', 'acchgq', 'cogsq', 'cogsy', 'cstkeq', 'doq', 'doy', 'dpq', 'dpy', 'dvpq',
        'dvpy', 'epsfiq', 'epsfiy', 'epsfxq', 'epsfxy', 'epspiq', 'epspiy',
        'epspxq', 'epspxy', 'epsx12', 'ibadjq', 'ibadjy', 'ibcomq', 'ibq',
        'iby', 'miiq', 'miiy', 'niq', 'niy', 'nopiq', 'nopiy', 'oiadpq', 'oiadpy',
        'oibdpq', 'opepsq', 'piq', 'piy', 'revtq', 'revty', 'saleq', 'saley', 'spiq',
        'spiy', 'txtq', 'txty', 'xidoq', 'xidoy', 'xintq', 'xiq', 'xiy', 'xoprq',
        'xopry', 'xsgaq']]

def drop_rows_with_half_missing_values(dataset):
    """
    data = [['a1', np.nan, np.nan],
        ['a2', 'b2', 'c2'],
        ['a3', np.nan, 'c3'], 
        [np.nan, 'b4', np.nan]]
    dataset = pd.DataFrame(data)
    """
    row_with_more_values_filled = (dataset.T.notna().mean() > 0.5)
    return dataset.loc[row_with_more_values_filled]
    

def drop_rows_where_SALEQ_ATQ_missing(dataset):
    """
    data = [['a1', 'a2', np.nan],
        ['a2', 'b2', 'c2'],
        ['a3', np.nan, 'c3'], 
        [np.nan, 'b4', 'c4']]
    dataset = pd.DataFrame(data, columns=['acoq', 'saleq', 'atq'])
    """
    filt = (dataset['saleq'].notna())
    dataset = dataset.loc[filt]
    filt2 = (dataset['atq'].notna())
    return dataset.loc[filt2]


def imputation_softimpute(dataset):
    #Need to do imputation for quarter with only information from prior quarters
    return SoftImpute().fit_transform(dataset)

def drop_rows_where_SALEQ_ATQ_zero(dataset):
    filt = (dataset['saleq'] != 0)
    dataset = dataset.loc[filt]
    filt2 = (dataset['atq'] != 0)
    return dataset.loc[filt2]

def exclude_quarters_with_no_accouncement_date(dataset):
    # Excluding quarters where YQ−0 has no announcement date and limiting the data to events between 1991 and 2017
    pass

def normalize_data():
    # Normalize balance sheet data by total assests and income and cash flow data by total sales
    # https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe
    pass

def minimize_stock_data_columns(stock_data):
    return stock_data[['TICKER', 'date', 'PRC']]

def drop_stock_nan_values(stock_data):
    filt = (stock_data['TICKER'].notna())
    stock_data = stock_data.loc[filt]
    filt2 = (stock_data['PRC'].notna())
    return stock_data.loc[filt2]

def make_price_positive(stock_data):
    stock_data['PRC'] = stock_data['PRC'].abs()
    return stock_data

def add_BHAR_column(stock_data):
    stock_data['BHAR'] = 0
    tickers = stock_data['TICKER'].unique()
    for tic in tickers:
        cond = stock_data['TICKER'] == tic
        ticker_stock_data = stock_data.loc[cond]
        stock_data.loc[cond, 'BHAR'] = ticker_stock_data['PRC'].pct_change(periods=1).shift(-1)
    return stock_data

def drop_BHAR_nan_values(stock_data):
    filt = (stock_data['BHAR'].notna())
    return stock_data.loc[filt]
    
def add_target_column(dataset, stock_data):
    pass


#dataset = pd.read_csv('data/fundamental_quarterly.csv', nrows=5)
#dataset = selecting_most_populated_columns(dataset)
#print(dataset)

#dataset.to_csv('data/most_populated_columns.csv', index=False)

#dataset = pd.read_csv('data/most_populated_columns.csv')
#dataset = drop_rows_with_half_missing_values(dataset)
#dataset.to_csv('data/most_populated_rows_and_columns.csv', index=False)

#dataset = pd.read_csv('data/most_populated_rows_and_columns.csv')
#dataset = drop_rows_where_SALEQ_ATQ_missing(dataset)
#dataset.to_csv('data/rows_SALEQ_ATQ_filled.csv', index=False)

#dataset = pd.read_csv('data/rows_SALEQ_ATQ_filled.csv')
#softimpute_np_arr = imputation_softimpute(dataset)
#dataset = pd.DataFrame(softimpute_np_arr, columns=['acoq', 'actq', 'ancq', 'aoq', 'apq', 'atq', 'capsq', 'ceqq', 'cheq','csh12q', 'cshoq', 'cshprq', 'cshpry', 'cstkq', 'dlcq', 'dlttq', 'dpactq', 'icaptq', 'invtq', 'lcoq', 'lctq', 'lltq', 'loq', 'lseq', 'ltmibq','ltq', 'mibq', 'mibtq', 'ppegtq', 'ppentq', 'pstknq', 'pstkq','pstkrq', 'rectq', 'req', 'seqq', 'tstkq', 'txditcq', 'txpq', 'wcapq', 'aolochy', 'apalchy', 'aqcy', 'capxy', 'chechy', 'dltisy', 'dltry','dpcy', 'dvy', 'esubcy', 'exrey', 'fiaoy', 'fincfy', 'fopoy', 'ibcy','intpny', 'invchy', 'ivacoy', 'ivchy', 'ivncfy', 'ivstchy', 'oancfy', 'prstkcy', 'recchy', 'sivy', 'sppivy', 'sstky', 'txdcy', 'xidocy', 'acchgq', 'cogsq', 'cogsy', 'cstkeq', 'doq', 'doy', 'dpq', 'dpy', 'dvpq','dvpy', 'epsfiq', 'epsfiy', 'epsfxq', 'epsfxy', 'epspiq', 'epspiy','epspxq', 'epspxy', 'epsx12', 'ibadjq', 'ibadjy', 'ibcomq', 'ibq','iby', 'miiq', 'miiy', 'niq', 'niy', 'nopiq', 'nopiy', 'oiadpq', 'oiadpy','oibdpq', 'opepsq', 'piq', 'piy', 'revtq', 'revty', 'saleq', 'saley', 'spiq','spiy', 'txtq', 'txty', 'xidoq', 'xidoy', 'xintq', 'xiq', 'xiy', 'xoprq','xopry', 'xsgaq'])
#dataset.to_csv('data/softimpute.csv', index=False)
#percent_missing = dataset.isnull().sum() * 100 / len(dataset)
#print(f'Percent Missing: {percent_missing}')

#dataset = pd.read_csv('data/softimpute.csv')
#dataset = drop_rows_where_SALEQ_ATQ_zero(dataset)
#dataset.to_csv('data/rows_SALEQ_ATQ_populated.csv', index=False)

#dataset = exclude_quarters_with_no_accouncement_date(dataset)
#dataset = normalize_data(dataset)

#print(f'Number of rows: {dataset.shape[0]}, Number of columns: {dataset.shape[1]}')



#Yahoo Finance Data

#yahoo_stock_data = yf.download(dataset['tic'].unique().tolist(), start="1983-01-01", end="2017-12-31")
#yahoo_stock_data.to_csv('data/yahoo_stock_data.csv', index=False)


#CRSP data
stock_data = pd.read_csv('data/crsp_data.csv', nrows = 20)
stock_data = minimize_stock_data_columns(stock_data)
stock_data = drop_stock_nan_values(stock_data)
stock_data = make_price_positive(stock_data)
stock_data = add_BHAR_column(stock_data)
stock_data = drop_BHAR_nan_values(stock_data)
print(stock_data)