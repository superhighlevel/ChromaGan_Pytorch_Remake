import torch
from src.discriminator import Discriminator
from src.colorization import Colorization
import configs.config as config
from utils.utils import deprocess, reconstruct, reconstruct_no
from torch.utils.data import DataLoader
from src.ColorizeDataloader import ColorizeDataLoader
import os
import sys
import numpy as np
import cv2


def sample_images(self, test_data, epoch):
    total_batch = int(test_data.size/config.BATCH_SIZE)
    for _ in range(total_batch):
            # load test data
            testL, _ ,  filelist, original, labimg_oritList  = test_data.generate_batch()

            # predict AB channels
            predAB, _  = self.colorizationModel.predict(np.tile(testL,[1,1,1,3]))

            # print results
            for i in range(config.BATCH_SIZE):
                    originalResult =  original[i]
                    height, width, channels = originalResult.shape

                    predictedAB = cv2.resize(deprocess(predAB[i]), (width,height))

                    labimg_ori =np.expand_dims(labimg_oritList[i], axis=2)
                    predResult = reconstruct(deprocess(labimg_ori), predictedAB, "epoch"+str(epoch)+"_"+filelist[i][:-5] )

def test(test_data):
    print('test')


def model(train_data, test_data, epochs, version = 0.0):
    # Create model folder if not exists
    model_name = config.MODEL_NAME + '_' + str(version) + '/'
    save_models_path = os.path.join(config.MODEL_DIR, model_name)
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    # Load training data and test data
    train_dataloader = ColorizeDataLoader(train_data)
    train_dataloader = DataLoader(train_dataloader, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    test_dataloader = ColorizeDataLoader(test_data)
    test_dataloader = DataLoader(test_dataloader, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    # Load the discriminator model and the colorization model   
    discriminator = Discriminator(input_size=224, in_channels=3).to(config.DEVICE)
    colorizationModel = Colorization(input_size=224, in_channels=3).to(config.DEVICE)

    # Real, Fake and Dummy for Discriminator
    positive_real = torch.ones(config.BATCH_SIZE, 1, dtype=np.float32).to(config.DEVICE)
    negative_real = -positive_real
    dummy_y = torch.zeros(config.BATCH_SIZE, 1, dtype=np.float32).to(config.DEVICE)

    # sample images after each epoch
    sample_images(test_data, config.EPORCHS)

    print('train')
    

def train():
    train_path = config.TRAIN_PATH
    test_path = config.TEST_PATH
    epochs = config.EPORCHS

    print('Start training...')
    print('-'*30)

    model(train_path, test_path, epochs)

    print('-'*30)
    print('Training done!')
    print('-'*30)

    print('Start testing...')
    print('-'*30)

    test(test_path)

    print('-'*30)
    print('Testing done!')
    print('-'*30)

def test_case_1():
    print('test case 1 start')
    positive_real = torch.ones(config.BATCH_SIZE, 1, dtype=np.float32).to(config.DEVICE)
    negative_real = -positive_real
    dummy_y = torch.zeros(config.BATCH_SIZE, 1, dtype=np.float32).to(config.DEVICE)
    print('test case 1 complete')
    

def test():
    print('test case 1')
    test_case_1()

class Trainer():
    def __init__(self, config):
        self.config = config
        self.train_data = config.TRAIN_PATH
        self.test_data = config.TEST_PATH
        self.epochs = config.EPORCHS
        self.version = config.VERSION

    def train(self):
        train(self.train_data, self.test_data, self.epochs, self.version)

    def test(self):
        test()


    
