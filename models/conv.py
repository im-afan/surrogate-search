from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2d(x)
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 5:
            x = x.view(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        #x = self.relu4(x)
        return x 
