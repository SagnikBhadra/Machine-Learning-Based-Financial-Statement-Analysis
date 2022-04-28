import models.rnn
import numpy as np
import torch
import torch.nn as nn


def generate_sequences(x, y, tickers):

	'''
	generates sequence data for rnn based on ticker

	'''
	
	x_sequences = []
	y_sequences = []
	i = 0
	while i < len(tickers) - 4:
		
		# if next 5 have same ticker then generate sequence
		if (tickers[i:i+5] == tickers[i]).all():
			x_seq = torch.unsqueeze(torch.from_numpy(x[i:i+4,:].astype(np.float32)), 0)
			y_seq = torch.tensor(y[i+4])

			x_sequences.append(x_seq)
			y_sequences.append(y_seq)
		
		i += 1

	return x_sequences, y_sequences


def load_data(path):

	'''
	loads data

	'''

	data = np.load(path, allow_pickle=True)

	tickers = data[:,0]
	x_train = data[:,2:-2]
	y_train = data[:,-1]

	return x_train, y_train, tickers



def main():

	# learning rate
	lr = .005
	num_epochs = 3
	train_path = './data/rnn_train.npy'
	val_path = './data/rnn_val.npy'

	# get device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# instantiate model
	model = models.rnn.RNN(device).to(device)

	# define loss and optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

	# load data
	x_train, y_train, tickers_train = load_data(train_path)
	x_val, y_val, tickers_val = load_data(val_path)

	# generate sequences
	x_sequences_train, y_sequences_train = generate_sequences(x_train, y_train, tickers_train)
	x_sequences_val, y_sequences_val = generate_sequences(x_val, y_val, tickers_val)

	# train loop
	train_losses = []

	for epoch in range(num_epochs):

		print(f'Epoch {epoch}')
		print('# Batches:', len(x_sequences_train))

		train_loss = 0

		for batch, (x_sequence, y_sequence) in enumerate(zip(x_sequences_train, y_sequences_train)):

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

