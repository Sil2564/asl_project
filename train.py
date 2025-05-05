import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import SimpleCNN  # Importa il modello dalla classe cnn_model.py

# Impostazioni per l'addestramento
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Funzione di addestramento
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
