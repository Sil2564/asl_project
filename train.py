import torch

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Debug stampa dimensioni batch
            print(f"Batch {i+1}: Immagini = {images.size()}, Etichette = {labels.size()}")

            # Inizializza i gradienti
            optimizer.zero_grad()

            # Passaggio forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation e ottimizzazione
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Stampa della loss media per epoca
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss media: {avg_loss:.4f}")
