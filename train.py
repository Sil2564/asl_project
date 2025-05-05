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
    model.train()  # Imposta il modello in modalità addestramento
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Azzerare i gradienti
            optimizer.zero_grad()
            
            # Passaggio in avanti
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calcolo dei gradienti
            loss.backward()
            
            # Ottimizzazione
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Ogni 100 mini-batch, stampiamo la perdita
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

# Avvio dell'addestramento
train(model, train_loader, criterion, optimizer, num_epochs)