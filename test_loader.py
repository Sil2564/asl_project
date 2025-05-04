from custom_dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

# Stampa un batch di immagini e etichette
for images, labels in train_loader:
    print(f"Batch immagini: {images.shape}")
    print(f"Batch etichette: {labels}")
    break  # Solo il primo batch
