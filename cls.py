import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Net()
    input = torch.randn((32, 256))
    output = model(input)
    print(output.shape)
