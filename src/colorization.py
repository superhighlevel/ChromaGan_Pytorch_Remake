import os
import warnings

import configs.config as config
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from utils.utils import *

from src.ColorizeDataloader import ColorizeDataLoader

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
                    in_channels, out_channels, kernel_size,
                    stride, padding, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size,
                    stride, padding, padding_mode='reflect'),
                nn.ReLU(0.2)
            )

    def forward(self, x):
        return self.conv(x)


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding),
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
            nn.Linear(512 * 7 * 7, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
        )

        self.fully_connected_2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 1000),
            nn.Softmax()
        )

        self.midlevel_features_1 = ConvBlock(512, 512, 3, 1, 1)
        self.midlevel_features_2 = ConvBlock(512, 256, 3, 1, 1)

        self.output_1 = SimpleConvBlock(512, 256, 1, 1, 0)
        self.output_2 = SimpleConvBlock(256, 128, 3, 1, 1)
        self.output_3 = SimpleConvBlock(128, 64, 3, 1, 1)
        self.output_4 = SimpleConvBlock(64, 64, 3, 1, 1)
        self.output_5 = SimpleConvBlock(64, 32, 3, 1, 1)
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
        global_features_class = self.flatten(global_features)

        global_features_class = self.fully_connected_2(global_features_class)

        # Mid-level Features
        mid_level_features = self.midlevel_features_1(x_model)
        mid_level_features = self.midlevel_features_2(mid_level_features)

        # fusion of (VGG16 + Mid-level) + (VGG16 + Global)
        fusion = torch.cat((mid_level_features, global_features2), dim=1)

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

        return output, global_features_class


# Testing function

def test_colorization():
    print('test_colorization')
    x = torch.randn((16, 1, 224, 224))
    # stack of copy of x
    x_3 = torch.cat([x, x, x], dim=1)
    # to device
    save_image(x, 'aaa.png')
    x_3 = x_3.to('cuda')
    colorization = Colorization(input_size=224).to('cuda')
    colorization.apply(initialize_weights)
    pred, _ = colorization(x_3)
    print(type(pred))
    print(pred[0].shape)
    save_image(pred[0][0], '111.png')
    save_image(pred[0][1], '222.png')
    x = x.to('cuda')
    pred_full = torch.cat([x, pred], dim=1)
    save_image(pred_full, '333.png')
    # images_path = config.OUTPUT_PATH + 'colred' + '.png'
    # pred = pred.permute(0, 2, 3, 1)
    # pred = pred.cpu().detach().numpy()
    # print(pred.shape)
    # pred[0] = cv2.cvtColor(pred[0], cv2.COLOR_Lab2RGB)
    # pred_a = pred[0]
    # print(pred_a.shape)
    # # pred_a = pred_a.transpose(1, 2, 0)
    # imageio.imwrite('filename.jpeg', pred_a)

    # im.save("your_file.jpeg")
    # save_image(pred[0], images_path)
    # print('test_colorization done')





def test_coloization_dataloader():
    print('test_coloization_dataloader')
    colorization = Colorization(input_size=224, in_channels=1).to('cuda')
    test_dataloader = ColorizeDataLoader(config.TEST_PATH)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=4)
    print('sampling images')
    for idx, (gray, real, real_2, _) in enumerate(tqdm(test_dataloader)):
        l_3 = np.tile(gray, [1, 3, 1, 1])
        l_3 = torch.from_numpy(l_3).to(config.DEVICE)
        colorization = Colorization(input_size=224, in_channels=1).to('cuda')
        colored, _ = colorization(l_3)
        colored = colored.detach()
        if not os.path.exists(config.OUTPUT_PATH):
            os.makedirs(config.OUTPUT_PATH)
        images_path = config.OUTPUT_PATH + str(idx) + '.png'
        print('real', real.shape)
        real_path = config.OUTPUT_PATH + str(idx) + '_real.png'
        save_image(colored, images_path)
        save_image(real, real_path)
        break
    print('test_coloization_dataloader done')


if __name__ == '__main__':
    test_colorization()
    test_coloization_dataloader()
