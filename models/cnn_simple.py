import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch



class CNN1(nn.Module):
    def __init__(self, output_type='classification'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*16*16, 64)
        
        if output_type == 'classification':
            self.fc_out = nn.Linear(64, 1)  # Binary
        else:
            self.fc_out = nn.Linear(64, 1)  # Regression

        self.output_type = output_type

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
        if self.output_type == 'classification':
            x = torch.sigmoid(x)
        return x