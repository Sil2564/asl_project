import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=29):  # Supponendo 29 lettere (A-Z escluso J/Z) + classi extra
        super(SimpleCNN, self).__init__()

        # Strato convoluzionale 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Strato convoluzionale 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Output dopo due pool da 2x2: 64x16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Cambia la dimensione in base alla tua uscita
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Aggiungi la riga di debug per vedere la forma dell'output
        print(x.shape)  # Aggiungi questa riga per vedere la dimensione

        # Appiattisci il tensore per passarlo al layer fully connected
        x = x.view(x.size(0), -1)  # Questo garantisce che il tensore venga appiattito correttamente
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
