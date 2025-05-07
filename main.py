import os
from custom_dataset import get_dataloaders
from cnn_model import SimpleCNN
from train import train, evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Parametri
batch_size = 64
num_epochs = 10
learning_rate = 0.001
image_size = 64
model_path = "modello_asl.pth"

# Dataset
train_loader, val_loader, test_loader = get_dataloaders(data_dir='dataset', batch_size=batch_size, image_size=image_size)

# Debug primo batch
for images, labels in train_loader:
    print(f"Dimensioni batch immagini: {images.shape}")
    print(f"Dimensioni batch etichette: {labels.shape}")
    break

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=29)
model.to(device)

# Perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Se il modello esiste, lo carichiamo invece di addestrare
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Modello caricato da '{model_path}'")
else:
    train(model, train_loader, criterion, optimizer, num_epochs, device)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modello salvato in '{model_path}'")

# Valutazione
evaluate(model, test_loader, criterion, device)
print("✅ Valutazione completata.")
