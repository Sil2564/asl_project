import torch

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Debug: stampa dimensioni
            if i < 5:  # Solo per i primi 5 batch
                print(f"Batch {i+1}: Immagini = {images.shape}, Etichette = {labels.shape}")

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
