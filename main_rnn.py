import models.rnn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RNN_Dataset(Dataset):

	def __init__(self, data_path):

		# load data
		data = np.load(data_path, allow_pickle=True)

		# some pre processing
		self.tickers = data[:,0]
		self.x = data[:,2:-2]
		self.y = data[:,-1]

		# generate sequences
		self.x_sequences, self.y_sequences = self.generate_sequences()



	def __len__(self):

		return len(self.x_sequences)

	def __getitem__(self, idx):

		return self.x_sequences[idx], self.y_sequences[idx]

	def get_tickers(self):

		return self.tickers


	def generate_sequences(self):

		'''
		generates sequence data for rnn based on ticker

		'''
		
		x_sequences = []
		y_sequences = []
		i = 0
		while i < len(self.tickers) - 4:
			
			# if next 5 have same ticker then generate sequence
			if (self.tickers[i:i+5] == self.tickers[i]).all():
				x_seq = torch.from_numpy(self.x[i:i+4,:].astype(np.float32))
				y_seq = torch.tensor(self.y[i+4])

				x_sequences.append(x_seq)
				y_sequences.append(y_seq)
			
			i += 1

		return x_sequences, y_sequences



def main():

	# learning rate
	lr = .005
	num_epochs = 5
	batch_size = 5
	train_path = './data/rnn_train_data_test.npy'#'./data/rnn_train.npy'
	val_path = './data/rnn_val.npy'

	# get device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# instantiate model
	model = models.rnn.RNN(device=device).to(device)

	# define loss and optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr) 


	# create datasets
	train_dataset = RNN_Dataset(train_path)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


	# train loop
	train_losses = []

	for epoch in range(num_epochs):

		print(f'Epoch {epoch}')
		print('# Batches:', len(train_dataloader))

		train_loss = 0

		for batch, (x_sequence, y_sequence) in enumerate(train_dataloader):

			if batch and batch % 1000 == 0:
				print(batch)

			# forward pass
			out = model.forward(x_sequence)
	
			# compute loss
			loss = 	criterion(out, y_sequence)

			# zero gradients
			optimizer.zero_grad()

			# compute gradients on loss
			loss.backward()

			# update weights
			optimizer.step()

			with torch.no_grad():
				train_loss += loss.numpy()

		print('Training Loss:', train_loss)
		train_losses.append(train_loss)
	
	
if __name__ == '__main__':

	main()

