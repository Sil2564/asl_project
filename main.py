from custom_dataset import get_dataloaders
from cnn_model import SimpleCNN
from train import train
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Percorso immagine per debug
img_path = r'C:\Users\busti\Desktop\asl_project\dataset\test\B\B1.jpg'
img = Image.open(img_path)
print(f"Tipo di immagine: {img.mode}")
img.show()

# Parametri
batch_size = 64
num_epochs = 10
learning_rate = 0.001
image_size = 32  # ⬅️ Modificato da 64 a 32

# Dataset
train_loader, val_loader, test_loader = get_dataloaders(data_dir='dataset', batch_size=batch_size, image_size=image_size)

# Debug primo batch
for images, labels in train_loader:
    print(f"Dimensioni batch immagini: {images.shape}")  # [64, 1, 32, 32] atteso
    print(f"Dimensioni batch etichette: {labels.shape}")  # [64]
    break

# Modello e addestramento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=29)  # ⬅️ Confermato num_classes=29
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Avvio training
train(model, train_loader, criterion, optimizer, num_epochs, device)
