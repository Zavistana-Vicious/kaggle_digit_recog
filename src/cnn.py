import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 8, 5, 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2),
        )

        self.full_c = torch.nn.Sequential(
            torch.nn.Linear(968, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.full_c(x)
        return x