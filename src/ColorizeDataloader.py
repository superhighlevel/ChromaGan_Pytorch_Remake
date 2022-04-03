import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from tqdm import tqdm


import configs.config as config


class ColorizeDataLoader(Dataset):
    def __init__(self, data_dir):

        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
