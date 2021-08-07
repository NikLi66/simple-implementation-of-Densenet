#python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def transform(mode):
    """
    Apply one of randomly cropping, rotation, changing brightness, changing contrast and horizontal flipping for each input image.
    """
    if mode == 'train':
        ComposedTransform=transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])
        return ComposedTransform
    elif mode == 'test':
        ComposedTransform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])
        return ComposedTransform



class Network(nn.Module):
    '''
    k is the grow rate of DenseNet, theta is the compression coefficient
    dense is a list which contain the depth of each layer
    theta is the coefficient used in compression/transition layer
    '''
    def __init__(self, k=12, dense=[16,16,16], theta=0.5):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels=3, out_channels=2*k, kernel_size=(3,3), padding=(1,1), bias=False)
        self.Pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        # Construct Dense Blocks and Transition Layers
        # Parameter delta is the coefficient of input channels
        Model = []
        in_channels = 2*k
        for i in range(len(dense)):
          Model.append(self._make_dense_block(in_channels, k, dense[i]))
          if i < len(dense)-1:
            in_channels += dense[i]*k
            out_channels = int(in_channels*theta)
            Model.append(self._make_transition_layer(in_channels, out_channels))
            in_channels = out_channels
        self.model = nn.Sequential(*Model)
        
        # Classifier
        # There is only 1 layer in this classifier.
        self.classifier = nn.Linear(1*1*(out_channels+dense[-1]*k), 10)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

    def _make_dense_block(self, in_channels, k, number_of_layers):
        layers = []
        for l in range(number_of_layers):
            layers.append(Bottleneck_Layer(in_channels + l*k, k))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_channels, out_channels):
        modules = []
        modules.append(Transition_Layer(in_channels, out_channels))
        return nn.Sequential(*modules)

    def forward(self, x):
        out = self.Conv(x)
        out = self.Pool(out)
        out = self.model(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class Bottleneck_Layer(nn.Module):
    """
    Each bottleneck layer consists of BN-ReLU-Conv1*1-BN-ReLU-Conv3*3.
    The Conv1*1 is to improve computation efficiency.
    The number of output channel of Conv1*1 is 4*k, which is followed the original paper.
    """
    def __init__(self, in_channels, k):
        super(Bottleneck_Layer, self).__init__()
        self.composite = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels=4*k, kernel_size=(1,1)),
                nn.Dropout(p=0.2),
                nn.BatchNorm2d(4*k),
                nn.ReLU(inplace=True),
                nn.Conv2d(4*k, out_channels=k, kernel_size=(3, 3), padding=(1,1)),
                nn.Dropout(p=0.2)
            )

    def forward(self, x):
        x_1 = self.composite(x)
        output = torch.cat([x, x_1], 1)         # Concatenation of input and output
        return output


class Transition_Layer(nn.Module):
    """
    Each transition layer consists of BN-Conv1*1-AvgPool.
    This is to improve model compactness.
    The parameter theta is the compression coefficient 
    """
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        output=self.transition(x)
        return output



net = Network()
lossFunc = nn.CrossEntropyLoss()

batch_size = 64
epochs = 150

