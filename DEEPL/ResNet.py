import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        
        # Première couche de convolution
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Seconde couche de convolution
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Fonction d’activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # on sauvegarde l’entrée originale
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Ajout de la connexion résiduelle
        out += identity
        out = self.relu(out)

        return out