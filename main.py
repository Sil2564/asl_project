from train import train
from cnn_model import SimpleCNN

# Impostazioni
num_epochs = 10

# Crea il modello
model = SimpleCNN(num_classes=10)  # 10 classi per il dataset MNIST

# Avvia il training
train(model, criterion, optimizer, num_epochs)
