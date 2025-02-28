import torch.nn as nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = self.conv5(x)

        return x
    
class WatermarkCNN(nn.Module):
    def __init__(self):
        super(WatermarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = self.conv5(x)

        return x


def count_parameters(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_in_MB = param_count * 4 / (1024 ** 2) 
    return param_size_in_MB

if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Total parameters: {count_parameters(model)}")