import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, device):
        super(RNN, self).__init__()
      
        self.device = device
        self.input_size = 121
        self.num_layers = 10
        self.hidden_size = 20
        self.output_size = 1

        
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
       
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):


        # initialize hidden state
        hidden_initial = torch.zeros(self.num_layers, x.shape[0],  self.hidden_size).to(self.device)

        out, _ = self.rnn(x, hidden_initial)

        # take last hidden state
        out = out[:, -1, :]

        # send through linear layer
        out = self.linear_layer(out)
  

        return out.flatten()

