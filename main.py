from cnn_model import SimpleCNN
from train import train
import torch
import torch.nn as nn
import torch.optim as optim

# Istanzia il modello
model = SimpleCNN()

# Definisci la funzione di perdita
criterion = nn.CrossEntropyLoss()

# Definisci l'ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Numero di epoche
num_epochs = 10

# Avvia il training
train(model, criterion, optimizer, num_epochs)

