import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='dataset', batch_size=64, image_size=64):
    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Scala di grigi (1 canale)
        transforms.Resize((image_size, image_size)),  # 👈 Resize dinamico in base al parametro
        transforms.ToTensor(),                        # Converte in tensore PyTorch
        transforms.Normalize((0.5,), (0.5,))          # Normalizza tra -1 e 1 (per immagini in scala di grigi)
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
