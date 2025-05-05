import torch
import torch.nn as nn

# Definizione della CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=26):  # Numero di classi nel dataset ASL (alfabeto inglese)
        super(SimpleCNN, self).__init__()

        # Strato convoluzionale 1 (modifica: ora accettiamo immagini RGB)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Strato convoluzionale 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Strato completamente connesso (fully connected)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Supponiamo che l'immagine di input sia 32x32
        self.fc2 = nn.Linear(128, num_classes)  # Il numero di classi è 26 (alfabeto ASL)

    def forward(self, x):
        # Passaggio attraverso il primo strato convoluzionale + attivazione ReLU
        x = self.pool(torch.relu(self.conv1(x)))

        # Passaggio attraverso il secondo strato convoluzionale + attivazione ReLU
        x = self.pool(torch.relu(self.conv2(x)))

        # Appiattimento (flatten) per passare dal formato 2D a 1D per la rete completamente connessa
        x = x.view(-1, 64 * 8 * 8)

        # Passaggio attraverso il primo strato completamente connesso
        x = torch.relu(self.fc1(x))

        # Passaggio attraverso l'ultimo strato completamente connesso (output)
        x = self.fc2(x)
        return x

