from custom_dataset import get_dataloaders
from cnn_model import SimpleCNN
from train import train, evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Parametri
batch_size = 64
num_epochs = 10
learning_rate = 0.001
image_size = 32  # Immagini ridimensionate a 32x32
num_classes = 29
model_save_path = "asl_cnn.pth"  # Percorso file per salvare il modello

# Caricamento dataset
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir='dataset',
    batch_size=batch_size,
    image_size=image_size
)

# Inizializzazione modello e setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Addestramento
train(model, train_loader, criterion, optimizer, num_epochs, device)

# Salvataggio modello
torch.save(model.state_dict(), model_save_path)
print(f"✅ Modello salvato in: {model_save_path}")

# Valutazione finale
evaluate(model, test_loader, criterion, device)
