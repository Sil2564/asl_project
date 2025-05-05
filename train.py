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

def evaluate(model, test_loader, criterion, device):
    model.eval()  # modalità valutazione
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():  # niente gradienti
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # classe predetta
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f"📊 Accuracy sul test set: {accuracy:.2f}%")
    print(f"📉 Loss media sul test set: {avg_loss:.4f}")
