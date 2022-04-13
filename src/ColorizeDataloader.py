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
        # Read the image
        grey_img, color_img, original, lab_img_ori = self.read_img(idx)
        # Transform the image to tensor
        grey_img = self.transform(grey_img)
        color_img = self.transform(color_img)
        # print(grey_img.shape)
        # print(color_img.shape)
        # original and lab_img_ori are not transformed since they are not used in the model
        return grey_img, color_img, original, lab_img_ori
    
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
        lab_img_ori = cv2.cvtColor(img_color, cv2.COLOR_BGR2Lab)

        # Return the grey image, ab image, original image, original lab image
        return (
            np.reshape(lab_img[:, :, 0], (self.img_size, self.img_size, 1)),
            #lab_img[:, :, 1:],
            #np.reshape(lab_img[:, :, 0], (1,self.img_size, self.img_size)),
            np.reshape(lab_img[:, :, 1:],(self.img_size, self.img_size,2)),
            img_color, 
            lab_img_ori)
    def generate_batch(self):
        batch = []
        labels = []
        filelist = []
        for i in range(config.batch_size):
            filename = os.path.join(self.color_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg = self.read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/255 # values between 0 and 1
        labels = np.asarray(labels)/255 # values between 0 and 1
        return batch, labels, filelist


def testing_colorize_dataloader():
    """
    Testing the dataloader
    """
    # Create the dataloader
    color_loader = ColorizeDataLoader(config.TRAIN_PATH)
    # Get the first image
    grey_img, color_img, original, lab_img_ori = color_loader[0]
    # Show the image
    print('grey_img: ', grey_img.shape)
    print('color_img: ', color_img.shape)
    print('original: ', original.shape)
    print('lab_img: ', lab_img_ori.shape)
    cv2.waitKey(0)

def checking_data_loader():
    """
    Checking the dataloader
    """
    # Create the dataloader
    print('Checking the dataloader')
    color_loader = ColorizeDataLoader(config.TRAIN_PATH)
    test_dataloader = DataLoader(
        color_loader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, drop_last=True)
    for idx, (grey_img, color_img, original, lab_img_ori) in enumerate(tqdm(test_dataloader)):
        continue
    print('Everything is fine')

if __name__ == '__main__':
    #checking_data_loader()
    testing_colorize_dataloader()
        
