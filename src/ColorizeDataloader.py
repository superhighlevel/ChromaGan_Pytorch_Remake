import glob
import os

import configs.config as config
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader



class ColorizeDataLoader(Dataset):
    """
    Data loader for the Colorization model
    :param color_path: path to the color images
    :param img_size: size of the image
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :param num_workers: number of workers
    """
    def __init__(self, color_path, img_size=224):
        # Raise error if the path is not a directory
        if not os.path.isdir(color_path):
            raise Exception("The path is not a directory")
        # Raise error if images size is not 224
        if img_size != 224:
            raise Exception("The image size must be 224")

        self.color_path = color_path
        self.img_size = img_size
        self.color_channels = 3
        self.gray_channels = 1
        self.data_color = []
        self.filelist = os.listdir(self.color_path)[:None]
        self.size = len(self.filelist)

        # Add '/' to the end of the path if it doesn't exist
        if self.color_path[-1] != '/':
            self.color_path += '/'
        
        # Get all the path of the color images
        for path_images_name in glob.glob(self.color_path + '*'):
            self.data_color.append(path_images_name)
        
        # Raise error if no data is found
        if len(self.data_color) == 0:
            print(self.color_path)
            raise Exception("Find no images in folder! Check your path", self.color_path)

    def __len__(self):
        return len(self.data_color)
    
    def __getitem__(self, idx):
        # Read the image``
        grey_img, color_img, original_images_shape = self.read_img(idx)

        grey_img = self.transform(grey_img)
        color_img = self.transform(color_img)
        # Return the resized image and the original shape
        return grey_img, color_img, original_images_shape
    def transform(self, img):
        """
        Transform function for the image. It converts the image to tensor
        :param img: image
        :return: tensor image
        """
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        return trans(img)

    def read_img(self, idx):
        """
        Read and covert the image to the required size
        :param idx: index of the image
        :return grey image, ab image, original image, lab image
        """
        # Read the image
        img_color_path = self.data_color[idx]
        img_color = cv2.imread(img_color_path)
        # Convert ting the image to the required size and convert to lab color space
        lab_img = cv2.cvtColor(
            cv2.resize(img_color, (self.img_size, self.img_size)),
            cv2.COLOR_BGR2Lab)
        # Get original images shape 
        original_shape = img_color.shape
        # print(original_shape)
        return (
            np.reshape(lab_img[:, :, 0], (self.img_size, self.img_size, 1)),
            np.reshape(lab_img[:, :, 1:],(self.img_size, self.img_size,2)),
            original_shape,
            )


def testing_colorize_dataloader():
    """
    Testing the dataloader
    """
    # Create the dataloader
    color_loader = ColorizeDataLoader(config.TRAIN_PATH)
    # Get the first image
    grey_img, color_img, original_images_shape = color_loader[0]
    # Show the image
    print('grey_img: ', grey_img.shape)
    print('color_img: ', color_img.shape)
    print('original_images_shape: ', original_images_shape)

def checking_data_loader():
    """
    Checking the dataloader
    """
    # Create the dataloader
    print('Checking the dataloader')
    # Please check your dataset
    houses = './dataset/houses/color_images'
    # color_loader = ColorizeDataLoader(config.TRAIN_PATH)
    color_loader = ColorizeDataLoader(houses)    
    test_dataloader = DataLoader(
        color_loader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, drop_last=True)
    for idx, (grey_img, color_img, _) in enumerate(tqdm(test_dataloader)):
        continue
    print('Everything is fine! Maybe :/ ?')

if __name__ == '__main__':
    #checking_data_loader()
    testing_colorize_dataloader()
        
