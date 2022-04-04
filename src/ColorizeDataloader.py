import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from tqdm import tqdm
import configs.config as config
import os
import configs.config_aa as config_aa
import cv2
import numpy as np
class ColorizeDataLoader(Dataset):
    def __init__(self, color_path, img_size = 64):
        self.color_path = color_path
        self.img_size = img_size
        self.color_channels = 3
        self.gray_channels = 1
        self.data_color = []
        self.data_gray = []

        if self.color_path[-1] != '/':
            self.color_path += '/'
        
        if self.gray_path[-1] != '/':
            self.gray_path += '/'

        for path_images_name in glob.glob(self.color_path + 'color_images/' + '*'):
            self.data_color.append(path_images_name)

        if len(self.data_color) == 0  or len(self.data_gray) == 0:
            raise Exception("Find no images in folder! Check your path")

        pass
    def __len__(self):
        return len(self.data_color)

    def read_img(self, idx):
        img_color_path= self.data_color[idx]
        img_color = cv2.imread(img_color_path)
        lab_img = cv2.cvtColor(
            cv2.resize(img_color, (self.img_size, self.img_size)), 
            cv2.COLOR_BGR2Lab)
        lab_img_ori = cv2.cvtColor(img_color, cv2.COLOR_BRG2Lab)
        return (
            np.reshape(lab_img[:,:,0], (self.img_size, self.img_size, 1)), 
            lab_img[:, :, 1:], 
            img_color, lab_img_ori[:,:,0])
    def transfrom(self, img):
        transform = transforms.Compose([
            transform.Normalize(),
            transforms.ToTensor(),
        ])
        return transform(img)

    def __getitem__(self, idx):
        grey_img, color_img, original,lab_img_ori = self.read_img(idx)
        grey_img = self.transfrom(grey_img)
        color_img = self.transfrom(color_img)
        lab_img_ori = self.transfrom(lab_img_ori)
        return grey_img, color_img, original, lab_img_ori




        
