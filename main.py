import numpy as np
import torch
import torch.nn as nn

from models import DNN, RNN, RandomForest

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load data

    #Set up model
    model = RNN().to(device)

    #Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    #Train the model
    


if __name__ == "__main__":
    main()