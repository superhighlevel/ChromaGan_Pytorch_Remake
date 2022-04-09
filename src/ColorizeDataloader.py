import glob
import os

import configs.config as config
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


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

    def read_img(self, idx):
        """
            Read and covert the image to the required size
            :param idx: index of the image
            :return: grey image, ab image, original image, lab image
        """
        # Read the image
        img_color_path = self.data_color[idx]
        img_color = cv2.imread(img_color_path)
        # Convert ting the image to the required size and convert to lab color space
        lab_img = cv2.cvtColor(
            cv2.resize(img_color, (self.img_size, self.img_size)),
            cv2.COLOR_BGR2Lab)
        lab_img_ori = cv2.cvtColor(img_color, cv2.COLOR_BGR2Lab)

        # Return the grey image, ab image, original image, original lab image
        return (
            np.reshape(lab_img[:, :, 0], (self.img_size, self.img_size, 1)),
            lab_img[:, :, 1:],
            img_color, 
            lab_img_ori)

    def transfrom(self, img):
        """
            Transform function for the image. It converts the image to tensor
            :param img: image
            :return: tensor image
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(img)

    def __getitem__(self, idx):
        # Read the image
        grey_img, color_img, original, lab_img_ori = self.read_img(idx)
        # Transform the image to tensor
        grey_img = self.transfrom(grey_img)
        color_img = self.transfrom(color_img)
        # original and lab_img_ori are not transformed since they are not used in the model
        return grey_img, color_img, original, lab_img_ori




        
