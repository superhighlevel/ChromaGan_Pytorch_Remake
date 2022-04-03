import torch
import torch.nn as nn
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")

class ConvBlock(nn.Module):
    """
        ConvBlock for the Colorization model
    """
    def __init__(
        self, in_channels, out_channels, 
        kernel_size, stride, padding, use_batchnorm=True):
        super().__init__()
        if use_batchnorm is False:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,kernel_size, 
                    stride, padding, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,kernel_size, 
                    stride, padding, padding_mode='reflect'),
                nn.ReLU(0.2)
            )
    
    def forward(self, x):
        return self.conv(x)
    
class Colorization(nn.Module):
    """
        The Colorization model that takes an input image and outputs a colorized image.
        The model is based on the VGG16 network.

    """
    
    def __init__(self, input_size, in_channels=3):
        super(Colorization, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.features.children())[:-8])


        self.global_features_1 = ConvBlock(512, 512, 3, 2, 1)
        self.global_features_2 = ConvBlock(512, 512, 3, 1, 1)

        self.global_features_3 = ConvBlock(512, 512, 3, 2, 1)
        self.global_features_4 = ConvBlock(512, 512, 3, 1, 1)

        self.flatten = nn.Flatten()

        self.fully_connected_1 = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
        )
        
        self.fully_connected_2 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 1000),
            nn.Softmax()
        )

        self.midlevel_features_1 = ConvBlock(512, 512, 3, 1, 1)
        self.midlevel_features_2 = ConvBlock(512, 256, 3, 1, 1)

        self.output_1 = ConvBlock(512, 256, 1, 1, 1, use_batchnorm=False)
        self.output_2 = ConvBlock(256, 128, 3, 1, 1, use_batchnorm=False)

        self.output_3 = ConvBlock(128, 64, 3, 1, 1, use_batchnorm=False)
        self.output_4 = ConvBlock(64, 64, 3, 1, 1, use_batchnorm=False)

        self.output_5 = ConvBlock(64, 32, 3, 1, 1, use_batchnorm=False)
        self.output_6 = nn.Sequential(
            nn.Conv2d(32, 2, 3, 1, 1),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        x_model = self.vgg(x)
        # Global features
        global_features = self.global_features_1(x_model)
        global_features = self.global_features_2(global_features)
        global_features = self.global_features_3(global_features)
        global_features = self.global_features_4(global_features)

        # Global features 2
        global_features2 = self.flatten(global_features)
        global_features2 = self.fully_connected_1(global_features2)
        global_features2 = global_features2.repeat(28, 28, 1, 1)
        global_features2 = global_features2.permute(2, 3, 0, 1)
        global_featuresClass = self.flatten(global_features)
        global_featuresClass = self.fully_connected_2(global_featuresClass)

        # Midlevel Features
        midlevel_features  = self.midlevel_features_1(x_model)
        midlevel_features = self.midlevel_features_2(midlevel_features)

        # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
        fusion = torch.cat((midlevel_features, global_features2), dim=1)

        # Output
        output = self.output_1(fusion)
        output = self.output_2(output)
        output = self.upsample(output)

        output = self.output_3(output)
        output = self.output_4(output)
        output = self.upsample(output)

        output = self.output_5(output)
        output = self.output_6(output)
        output = self.upsample(output)

        return output, global_featuresClass

def test_colorization():
    print('test_colorization')
    x = torch.randn((8, 1, 224, 224))
    # stack of copy of x
    x = torch.cat([x, x, x], dim=1)
    # to device
    x = x.to('cuda')
    colorization = Colorization(input_size=224, in_channels=3).to('cuda')
    pred = colorization(x)
    print(pred[0].shape)
    print(pred[1].shape)
    print('test_colorization done')

if __name__ == '__main__':
    test_colorization()
        


