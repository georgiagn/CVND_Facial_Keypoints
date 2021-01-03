## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # The output Tensor for one image will have the dimensions: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer with kernel_size=2, stride=2
        # Output tensor size: (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)

        # dropout with p=0.1
        self.drop1 = nn.Dropout(p=0.1)
 
        # 32 input channels, 64 output channels/feature maps, 3x3 square convolution kernel
        # Output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # The output Tensor for one image will have the dimensions: (64, 108, 108)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Output tensor size: (64, 54, 54)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.2
        self.drop2 = nn.Dropout(p=0.2)

        # 64 input image channels, 128 output channels/feature maps, 3x3 square convolution kernel
        # Output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # The output Tensor for one image will have the dimensions: (128, 52, 52)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Output tensor size: (128, 26, 26)
        self.pool3 = nn.MaxPool2d(2, 2)

        # dropout with p=0.3
        self.drop3 = nn.Dropout(p=0.3)

        # 128 input image channels, 256 output channels/feature maps, 3x3 square convolution kernel
        # Output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # The output Tensor for one image will have the dimensions: (256, 24, 24)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # Output tensor size: (256, 12, 12)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.4
        self.drop4= nn.Dropout(p=0.4)

        # 256 outputs * the 12x12 filtered/pooled map size
        self.fc1 = nn.Linear(256*12*12, 1000)

        # dropout with p=0.5
        self.fc1_drop= nn.Dropout(p=0.5)
       
        # Another fully connected layer
        self.fc2 = nn.Linear(1000, 1000)

        # dropout with p=0.6
        self.fc2_drop= nn.Dropout(p=0.6)

        # Finally, create 136 output channels (2 for each of the 68 keypoint (x, y) pairs)
        self.fc3 = nn.Linear(1000, 136)

               

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model

        # four conv/relu + pool + dropout layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x) 
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x) 
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
                   
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
