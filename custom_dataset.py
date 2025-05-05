import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloaders(data_dir='dataset', batch_size=32, image_size=64):
    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Ridimensiona tutte le immagini
        transforms.ToTensor(),  # Converte in tensore PyTorch
        transforms.Normalize((0.5,), (0.5,))  # Normalizzazione tra -1 e 1
    ])

    # Funzione per garantire che le immagini siano RGB
    def rgb_loader(path):
        img = Image.open(path)
        if img.mode != 'RGB':  # Se l'immagine non è RGB, convertila
            img = img.convert('RGB')
        return img

    # Crea dataset con il loader personalizzato che assicura l'uso di immagini RGB
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform, loader=rgb_loader)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform, loader=rgb_loader)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform, loader=rgb_loader)

    # Crea DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
