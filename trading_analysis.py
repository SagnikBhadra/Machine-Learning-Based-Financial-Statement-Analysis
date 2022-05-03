import torch
import numpy as np
import models.rnn
import models.dnn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

class RNN_Dataset(Dataset):

	def __init__(self, data_path):

		# load data
		data = np.load(data_path, allow_pickle=True)

		# some pre processing
		self.tickers = data[:,0]
		self.x = data[:,2:-2]
		self.y = data[:,-1]
		self.dates = data[:,1].astype(int).astype(str)

		# generate sequences
		self.x_sequences, self.y_sequences, self.dates = self.generate_sequences()



	def __len__(self):

		return len(self.x_sequences)

	def __getitem__(self, idx):

		return self.x_sequences[idx], self.y_sequences[idx], self.dates[idx]

	def get_tickers(self):

		return self.tickers


	def generate_sequences(self):

		'''
		generates sequence data for rnn based on ticker

		'''
		
		x_sequences, y_sequences, dates = [], [], []
		i = 0
		while i < len(self.tickers) - 4:
			
			# if next 5 have same ticker then generate sequence
			if (self.tickers[i:i+5] == self.tickers[i]).all():
				x_seq = torch.from_numpy(self.x[i:i+4,:].astype(np.float32))
				y_seq = torch.tensor(self.y[i+4])

				x_sequences.append(x_seq)
				y_sequences.append(y_seq)
				dates.append(self.dates[i+4])
			
			i += 1

		return x_sequences, y_sequences, dates


def prepare_data(data):

	tickers = data[:,0]
	x = data[:,2:-2]
	y = data[:,-1]
	dates = data[:,1].astype(int).astype(str)

	x_out, y_out , dates_out = [], [], []
	i = 0
	while i < len(tickers) - 4:
		# if next 5 have same ticker then generate sequence
		if (tickers[i:i+5] == tickers[i]).all():
			x_out.append(list(x[i:i+4,:].flatten()))

			y_out.append(y[i+4])
			dates_out.append(int(dates[i+4][:4]))
		
		i += 1

	x_out = np.array(x_out)

	return x_out, np.array(y_out), dates_out

def trading_analysis():

	# get device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	val_path = './data/rnn_val.npy'

	# load data
	data = np.load(val_path, allow_pickle=True)

	val_dataset_rnn = RNN_Dataset(val_path)
	val_dataloader_rnn = DataLoader(val_dataset_rnn, batch_size=len(val_dataset_rnn), shuffle=False)
	
	# prepare data for other models
	x_other, y_other, dates_other = prepare_data(data)


	# load models
	models_list = []

	# load rnn model
	model_rnn = models.rnn.RNN(device=device).to(device)
	model_path = './trained_models/rnn_model_mse_loss.pt'
	model_rnn.load_state_dict(torch.load(model_path))
	model_rnn.eval()
	models_list.append(model_rnn)

	model_path = './trained_models_sagnik/dnn_model_mae.pt'
	model_dnn = models.dnn.DNN().to(device)
	model_dnn.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
	model_dnn.eval()
	models_list.append(model_dnn)

	#model_path = './trained_models_sagnik/lasso_model_mse.sav'
	#model = pickle.load(open(model_path, 'rb'))
	#print(model)
	#exit()



	for i, model in enumerate(models_list):

		print(model)
		model_return_dict = {}
		if i == 0:

			# rnn

			with torch.no_grad():
				for batch, (x_sequence, y_truth, dates) in enumerate(val_dataloader_rnn):

					y_pred = model.forward(x_sequence)

					pos_mask = y_truth*y_pred > 0
					y_truth[pos_mask] = torch.abs(y_truth[pos_mask])
					y_truth[~pos_mask] = -torch.abs(y_truth[~pos_mask])

					for val, date in zip(y_truth, dates):
						date_mod = date[:6]
						val = float(val.numpy())
						if int(date_mod) in model_return_dict:
							model_return_dict[int(date_mod)] += val
						else:
							model_return_dict[int(date_mod)] = val


					sorted_x = [str(k)[:6] for k,v in sorted(model_return_dict.items())]
					sorted_y = [v for k,v in sorted(model_return_dict.items())]
					plt.plot(sorted_x, np.cumsum(sorted_y))
		elif i == 1:

			# dnn
			y_pred = model(torch.from_numpy(x_other.astype(np.float32))).flatten().detach().numpy()
			y_truth = y_pred.copy()

			pos_mask = y_pred*y_other > 0
			y_truth[pos_mask] = np.abs(y_truth[pos_mask])
			y_truth[~pos_mask] = -np.abs(y_truth[~pos_mask])

			for val, date in zip(y_truth, dates):
				date_mod = date[:6]

				if int(date_mod) in model_return_dict:
					model_return_dict[int(date_mod)] += val
				else:
					model_return_dict[int(date_mod)] = val

			sorted_x = [str(k)[:6] for k,v in sorted(model_return_dict.items())]
			sorted_y = [v for k,v in sorted(model_return_dict.items())]
			plt.plot(sorted_x, np.cumsum(sorted_y))

			



	plt.show()




if __name__ == '__main__':

	trading_analysis()
