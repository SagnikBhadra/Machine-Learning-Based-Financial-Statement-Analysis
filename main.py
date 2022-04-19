import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import DNN, RNN, RandomForest
from clean_data import selecting_most_populated_columns, drop_rows_with_half_missing_values, drop_rows_where_SALEQ_ATQ_missing, imputation_softimpute, drop_rows_where_SALEQ_ATQ_zero, exclude_quarters_with_no_accouncement_date, normalize_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load data
    dataset = pd.read_csv('data/cleaned_data.csv')

    #Set up model
    model = RNN().to(device)

    #Initialize loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    #Train the model
    


if __name__ == "__main__":
    main()