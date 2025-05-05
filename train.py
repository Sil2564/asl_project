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

# Trasformazioni per il dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Assicura che le immagini siano in scala di grigi
    transforms.Resize((32, 32)),  # Ridimensiona le immagini a 32x32
    transforms.ToTensor(),  # Converte l'immagine in un tensore
])

# Caricamento del dataset (esempio con MNIST, se hai un altro dataset, fammi sapere)
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Definire il modello e il dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)  # 10 classi per MNIST

# Funzione di perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Funzione di addestramento
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Aggiungi questa riga per stampare le dimensioni
            print(f"Batch {i+1}: Input batch size = {images.size(0)}, Labels batch size = {labels.size(0)}")
            
            # Calcolo della perdita (loss)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Aggiorna i pesi del modello
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Avvio dell'addestramento
train(model, train_loader, criterion, optimizer, num_epochs)