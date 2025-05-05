from custom_dataset import get_dataloaders
from cnn_model import SimpleCNN
from train import train
import torch
import torch.nn as nn
import torch.optim as optim

# Ottieni i DataLoader
train_loader, val_loader, test_loader = get_dataloaders(data_dir='dataset', batch_size=64, image_size=64)

# Istanzia il modello
model = SimpleCNN()

# Definisci la funzione di perdita
criterion = nn.CrossEntropyLoss()

# Definisci l'ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Numero di epoche
num_epochs = 10

# Avvia il training
train(model, train_loader, criterion, optimizer, num_epochs)
