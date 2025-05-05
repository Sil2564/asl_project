import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='dataset', batch_size=32, image_size=64):
    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Ridimensiona tutte le immagini
        transforms.Grayscale(num_output_channels=3),  # Forza la conversione in RGB
        transforms.ToTensor(),  # Converte in tensore PyTorch
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizzazione tra -1 e 1
    ])

    # Crea dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Crea DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
