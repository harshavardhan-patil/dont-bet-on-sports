import torch.nn as nn

class NFLPredictor(nn.Module):
    def __init__(self, input_dim):
        super(NFLPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # Dropout and Layer Normalization
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(128)

        # Activation function
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        # Layer 1
        x1 = self.activation(self.norm1(self.layer1(x)))
        x1 = self.dropout(x1)

        # Layer 2 
        x2 = self.activation(self.norm2(self.layer2(x1)))
        x2 = self.dropout(x2)

        # Layer 3
        x3 = self.activation(self.norm3(self.layer3(x2)))
        x3 = self.dropout(x3)

        # Layer 4
        x4 = self.activation(self.layer4(x3))

        # Output layer
        output = self.output(x4)
        return output
