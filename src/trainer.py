import json
import os
import pandas as pd

import configs.config as config
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *

from src.colorization import Colorization
from src.ColorizeDataloader import ColorizeDataLoader
from src.discriminator import Discriminator


def model(train_data, test_data, epochs, version=0.0):
    """
    Create the model and train it for the given epochs
    :param train_data: train data
    :param test_data: test data
    :param epochs: number of epochs
    :param version: version of the model
    """
    # data loader
    train_dataloader = ColorizeDataLoader(train_data)
    train_dataloader = DataLoader(
        train_dataloader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, drop_last=True)

    test_dataloader = ColorizeDataLoader(test_data)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, drop_last=True)

    # Load the discriminator model and the colorization model   
    discriminator = Discriminator(input_size=224).to(config.DEVICE)
    discriminator.apply(initialize_weights)
    colorization_model = Colorization(input_size=224).to(config.DEVICE)
    vgg_model_f = models.vgg16(pretrained=True).to(config.DEVICE)

    positive_real = torch.ones(size=(config.BATCH_SIZE, 1, 28, 28), requires_grad=True).to(config.DEVICE)
    negative_real = (-positive_real).to(config.DEVICE)

    optimizer_g = Adam(
        list(colorization_model.parameters()),
        lr=config.LR,
        betas=(0.5, 0.999),
    )

    optimizer_d = Adam(
        list(discriminator.parameters()), lr=config.LR,
        betas=(0.5, 0.999))

    # init loss function
    KKLDivergence = nn.KLDivLoss()
    MSE = nn.MSELoss()

    for epoch in range(epochs):
        print(f'EPOCH {epoch} / {epochs}')
        print('-' * 30)

        for idx, (trainL, trainAB, _) in enumerate(tqdm(train_dataloader)):
            trainL = trainL.to(config.DEVICE)
            trainAB = trainAB.to(config.DEVICE)

            l_3 = torch.cat([trainL, trainL, trainL], dim=1)
            pred_class_vgg = F.softmax(vgg_model_f(l_3))
            # ----------------- Train the generator -----------------
            optimizer_g.zero_grad()
            pred_AB, pred_class_c = colorization_model(l_3)
            pred_LAB_C = torch.cat([trainL, pred_AB], dim=1)
            with torch.no_grad():
                dis_C = discriminator(pred_LAB_C)
            dis_C = dis_C.mean()
            KLD_loss = KKLDivergence(pred_class_c, pred_class_vgg.detach().float()) * 0.003
            MSE_loss = MSE(pred_AB, trainAB) * 10
            W_loss = wasserstein_loss(dis_C, True) * 0.1
            g_loss = KLD_loss + MSE_loss + W_loss

            # ----------------- Train the discriminator -----------------
            for param in discriminator.parameters():
                param.requires_grad = True
            optimizer_d.zero_grad()
            pred_LAB_D = torch.cat([trainL, pred_AB], dim=1)
            dis_pred = discriminator(pred_LAB_D)
            dis_pred = dis_pred.mean()

            true_LAB_D = torch.cat([trainL, trainAB], dim=1)
            dis_true = discriminator(true_LAB_D)
            dis_true = dis_true.mean()
            
            weights = torch.randn((trainAB.size(0),1,1,1), device=config.DEVICE)
            averaged_samples = (weights * trainAB) + ((1 - weights) * pred_AB)
            averaged_samples = torch.autograd.Variable(averaged_samples, requires_grad=True)
            avg_img = torch.cat([trainL, averaged_samples], dim=1)
            dis_avg = discriminator(avg_img)

            W_loss_true = wasserstein_loss(dis_true, False)
            W_loss_pred = wasserstein_loss(dis_pred, True)
            gp_loss_avg = partial_gp_loss(dis_avg, averaged_samples, config.GRADIENT_PENALTY_WEIGHT)
            d_loss = W_loss_true + W_loss_pred + gp_loss_avg
            with torch.autograd.set_detect_anomaly(True):
                g_loss.backward(retain_graph=True)
                d_loss.backward()
                optimizer_g.step()
                optimizer_d.step()
            # ----------------- Log the trainning process  -----------------
            if config.CHECK_PER!=-1:
                if idx % config.CHECK_PER == 0:
                    print('\n')
                    print(f"Epoch {epoch} - Batch {idx} - Loss G: {g_loss} - Loss D: {d_loss}")
    # create MODEL_DIR if not exist
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    torch.save(discriminator.state_dict(), config.MODEL_D)
    torch.save(colorization_model.state_dict(), config.MODEL_C)



def train():
    train_path = config.TRAIN_PATH
    test_path = config.TEST_PATH
    epochs = config.EPOCHS

    print('Start training...')
    print('-' * 30)

    model(train_path, test_path, epochs)

    print('-' * 30)
    print('Training done!')
    print('-' * 30)

    print('Start testing...')
    print('-' * 30)

    test()
    
    print('Testing done!')

    print('-' * 30)
    print('All done!')


def sample_images(test_data, colorizationModel):
    """
    Sample images after training
        :param test_data: test data
        :param colorizationModel: colorization model
    """
    print('Sampling images')
    for idx, (gray, ori_ab, _) in enumerate(tqdm(test_data)):
        l_3 = torch.cat([gray, gray, gray], dim=1).to(config.DEVICE)
        # torch required no grad
        with torch.no_grad():
            colored, _ = colorizationModel(l_3)

        gray = gray.detach().cpu().numpy()
        ori_ab = ori_ab.detach().cpu().numpy()
        colored = colored.detach().cpu().numpy()
        for i in range(config.BATCH_SIZE):
            original_result_red = reconstruct(deprocess(gray)[i], deprocess(colored)[i])
            #print('originalResult_red shape: ', original_result_red.shape)
            cv2.imwrite(config.OUTPUT_PATH + str(idx) + '.png', original_result_red)
    print('Sampling images done')


def test():
    """
    Test the model
    """
    path = config.MODEL_C
    test_dataloader = ColorizeDataLoader(config.TEST_PATH)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, drop_last=True)
    # test dataloader working correctly
    for idx, (grey_img, color_img, original_images_shape) in enumerate(test_dataloader): 
        print(f"{idx} / {len(test_dataloader)}")
        print(f"gray shape: {grey_img.shape}")
        print(f"ori_ab shape: {color_img.shape}")
        print(f"{original_images_shape}")
        break
    colorizationModel = Colorization(input_size=224).to(config.DEVICE)
    if config.DEVICE == 'cpu':
        colorizationModel.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        colorizationModel.load_state_dict(torch.load(path))
    colorizationModel.eval()
    sample_images(test_dataloader, colorizationModel)

def test_case_1(device):
    print('test case 1 start')
    positive_real = torch.ones(size=(config.BATCH_SIZE, 1))
    negative_real = -positive_real
    dummy_y = torch.zeros(size=(config.BATCH_SIZE, 1))
    print(positive_real.shape)
    print(negative_real.shape)
    print(dummy_y.shape)
    print('test case 1 complete')


def run_all_test_case(device):
    print('test case 1')
    test_case_1(device)


class Trainer:
    def __init__(self):
        self.device =config.DEVICE

    @staticmethod
    def train():
        train()


class Tester:
    def __init__(self):
        self.device =config.DEVICE

    def test(self):
        test()
