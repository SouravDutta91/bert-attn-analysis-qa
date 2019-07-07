# import libraries

import torch
import torch.nn as nn
import torch.optim as optim


# LSTM autoencoder class

class LSTM(nn.Module):
    
    def __init__(self, input_dim, latent_dim, num_layers):
        
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)
        self.decoder = nn.LSTM(self.latent_dim, self.input_dim, self.num_layers)
    
    def forward(self, input):
	
	# encoder
        _, (last_hidden, _) = self.encoder(input)
        encoded = last_hidden.repeat(input.shape)
        
	# decoder
        y, _ = self.decoder(encoded)
        return torch.squeeze(y)


# print epoch, loss, and prediction

def print_results(epoch, loss, prediction):
    
    print()
    print('Epoch: %d, Loss: %f, Prediction: %s' % (epoch, loss, prediction.data.cpu().numpy()))


# call function

def autoencoder(input_data, input_dim=1, latent_dim=20, num_layers=1, verbose=0):
    
    model = LSTM(input_dim=input_dim, latent_dim=latent_dim, num_layers=num_layers)
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    y = torch.Tensor(input_data)
    x = y.view(len(y), 1, -1)

    count = 0
    loss = 100

    while loss > 0.01:

        count += 1

        y_pred = model(x)
        optimizer.zero_grad()
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        
        if verbose == 1:
            print_results(count, loss, y_pred)


# example to call autoencoder() function
'''
def __main__():
    
    input_dim = 1
    latent_dim = 20
    num_layers = 1
    
    input_data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    autoencoder(input_data, input_dim, latent_dim, num_layers, verbose=1)

__main__()
'''
