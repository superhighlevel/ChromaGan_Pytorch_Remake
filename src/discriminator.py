import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Convolutional block for the Colorization model
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: kernel size
    :param stride: stride
    :param padding: padding
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    """
    Discriminator for the Colorization model
    :param input_size: size of the input image
    :param in_channels: number of input channels
    """
    def __init__(self, input_size, in_channels=3):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 4, 2, 1), #(112, 112, 64)
            nn.LeakyReLU(0.2)
        ) 
        self.conv_2 = ConvBlock(64, 128, 4, 2, 1) # (56, 56, 128)
        self.conv_3 = ConvBlock(128, 256, 4, 2, 1) # (28, 28, 256)
        self.conv_4 = ConvBlock(256, 512, 3, 1, 1) # (28, 28, 512)
        self.conv_5 = ConvBlock(512, 1, 3, 1, 1) # (28, 28, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return torch.sigmoid(x)


def test_discriminator():
    """
    Test the discriminator on a single image
    """
    print('test_discriminator')
    x = torch.randn((1, 3, 224, 224)).to('cuda')
    discriminator = Discriminator(input_size=224).to('cuda')
    pred = discriminator(x)
    print(pred.shape)
    print('test_discriminator done')

if __name__ == '__main__':
    test_discriminator()