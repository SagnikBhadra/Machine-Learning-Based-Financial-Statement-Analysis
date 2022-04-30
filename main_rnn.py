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



def train(
	lr=.001,
	num_epochs=5,
	batch_size=128,
):

	print()

	# paths
	train_path = './data/rnn_train.npy'
	val_path = './data/rnn_val.npy'

	# get device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# instantiate model
	model = models.rnn.RNN(device=device).to(device)

	# define loss and optimizer
	criterion = nn.MSELoss(reduction='sum')
	optimizer = torch.optim.RMSprop(model.parameters(), lr=lr) 


	# create datasets
	print('Loading Train Data...')
	train_dataset = RNN_Dataset(train_path)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	print('Loading Validation Data...')
	val_dataset = RNN_Dataset(val_path)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


	# train loop
	train_losses = []

	for epoch in range(num_epochs):

		print('-----------------------------')
		print(f'Epoch {epoch+1}')
		print('# Batches:', len(train_dataloader))

		train_loss = 0
		model.train()

		for batch, (x_sequence, y_sequence) in enumerate(train_dataloader):

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

		with torch.no_grad():
			train_loss /= len(train_dataset)

		print('Training Loss:', round(train_loss,4))
		train_losses.append(train_loss)


		# compute val loss
		model.eval()
		val_loss = 0
		with torch.no_grad():
			for batch, (x_sequence, y_sequence) in enumerate(val_dataloader):
				out = model.forward(x_sequence)
				loss = 	criterion(out, y_sequence)
				val_loss += loss.numpy()
			val_loss /= len(val_dataset)
			print('Validation Loss:', round(val_loss,4))

	# save model
	torch.save(model.state_dict(), './trained_models/rnn_model_rmse.pt')


def validate():

	# paths
	model_path = './trained_models/rnn_model_mse_loss.pt'
	train_path = './data/rnn_train.npy'
	val_path = './data/rnn_val.npy'

	# get device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# load
	model = models.rnn.RNN(device=device).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()


	# create datasets
	#print('Loading Train Data...')
	#train_dataset = RNN_Dataset(train_path)
	#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	print('Loading Validation Data...')
	val_dataset = RNN_Dataset(val_path)
	val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

	with torch.no_grad():
		for batch, (x_sequence, y_sequence) in enumerate(val_dataloader):

			out = model.forward(x_sequence)

			# evaluate metrics
			mse = nn.MSELoss()(y_sequence, out)
			rmse = torch.sqrt(mse)
			mae = nn.L1Loss()(y_sequence, out)
			medae = torch.median(nn.L1Loss(reduction='none')(y_sequence, out))

		print('MSE:', mse.numpy())
		print('RMSE:', rmse.numpy())
		print('MAE:', mae.numpy())
		print('MedAE:', medae.numpy())


	
	
if __name__ == '__main__':

	#train()
	validate()

