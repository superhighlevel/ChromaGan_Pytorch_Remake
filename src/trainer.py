import json
import os
import sys

import configs.config as config
import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from utils.utils import *
from utils.utils import deprocess, reconstruct, reconstruct_no

from src.colorization import Colorization
from src.ColorizeDataloader import ColorizeDataLoader
from src.discriminator import Discriminator


def sample_images(test_data, colorizationModel, epoch):
    print('sampling images')
    for idx, (gray, _, _, _) in enumerate(tqdm(test_data)):
        l_3 = torch.cat([gray, gray, gray], dim=1).to(config.DEVICE)
        colored, _ = colorizationModel(l_3)
        gray = gray.to(config.DEVICE)
        colored = torch.cat([gray, colored], dim=1)
        if not os.path.exists(config.OUTPUT_PATH):
            os.makedirs(config.OUTPUT_PATH)
        images_path = config.OUTPUT_PATH + str(idx) + '.png'
        save_image(colored, images_path)


def test():
    path = config.MODEL_PATH + 'color.pt'
    test_dataloader = ColorizeDataLoader(config.TEST_PATH)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=2)
    colorizationModel = Colorization(input_size=224, in_channels=1).to(config.DEVICE)
    colorizationModel.load_state_dict(torch.load(path))
    epoch = 1
    sample_images(test_dataloader, colorizationModel, epoch)


def model(train_data, test_data, epochs, version=0.0):
    # Create model folder if not exists
    model_name = config.MODEL_NAME + '_' + str(version) + '/'
    save_models_path = os.path.join(config.MODEL_DIR, model_name)
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    # Load training data and test data
    train_dataloader = ColorizeDataLoader(train_data)
    train_dataloader = DataLoader(
        train_dataloader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2)

    test_dataloader = ColorizeDataLoader(test_data)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2)

    # Load the discriminator model and the colorization model   
    discriminator = Discriminator(input_size=224).to(config.DEVICE)
    discriminator.apply(initialize_weights)
    colorization_model = Colorization(input_size=224, in_channels=1).to(config.DEVICE)
    colorization_model.apply(initialize_weights)
    vgg_model_f = models.vgg16(pretrained=True).to(config.DEVICE)

    # Real, Fake and Dummy for Discriminator
    #     positive_real = torch.ones(size = (BATCH_SIZE, 1)).to(DEVICE)
    #     negative_real = (-positive_real).to(DEVICE)
    #     dummy_y = torch.zeros(size = (BATCH_SIZE, 1)).to(DEVICE)
    positive_real = torch.ones(size=(config.BATCH_SIZE, 1, 28, 28), requires_grad=True).to(config.DEVICE)
    negative_real = (-positive_real).to(config.DEVICE)

    optimizer_g = Adam(
        list(colorization_model.parameters()),
        lr=config.LR,
        betas=(0.5, 0.999),
    )

    optimizer_d = Adam(
        list(discriminator.parameters()), lr=config.LR,
        betas=(0.9, 0.999))

    color_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    losses = {
        'Epoch': [],
        'mse_loss_gen': [], 'kld_loss_gen': [], 'wl_loss_gen': [],
        'loss_gen': [],
        'Loss_D_Fake': [], 'Loss_D_Real': [], 'Loss_D_avg': [],
        'Loss_D': []}
    for epoch in range(epochs):
        print(f'EPORCH {epoch} / {epochs}')
        losses['Epoch'].append(epoch)
        print('-' * 30)
        for idx, (trainL, trainAB, _, _) in enumerate(tqdm(train_dataloader)):
            l_3 = torch.cat([trainL, trainL, trainL], dim=1).to(config.DEVICE)
            predict_vgg = vgg_model_f(l_3)

            # Train the Generator

            l_3 = l_3.to(config.DEVICE)

            #             optimizer_g.zero_grad()
            predAB, pred_class = colorization_model(l_3)
            predAB = predAB.to(config.DEVICE)
            trainAB = trainAB.to(config.DEVICE)
            trainL = trainL.to(config.DEVICE)
            pred_lab = torch.cat([trainL, predAB], dim=1).to(config.DEVICE)
            disc_pred = discriminator(pred_lab)

            pred_class = pred_class.to(config.DEVICE)
            predict_vgg = predict_vgg.to(config.DEVICE)

            # Loss genrenator 
            mse_loss_gen = torch.nn.functional.mse_loss(predAB, trainAB)
            kld_loss_gen = torch.nn.functional.kl_div(pred_class, predict_vgg) * 0.003
            mal_gen = disc_pred * positive_real
            mal_gen = torch.autograd.Variable(mal_gen, requires_grad=True)
            wl_loss_gen = wasserstein_loss(mal_gen) * -0.1
            loss_gen = (mse_loss_gen + kld_loss_gen + wl_loss_gen)

            # Backpropagation
            optimizer_g.zero_grad()
            loss_gen.backward(retain_graph=True)
            optimizer_g.step()

            # append losses
            losses['mse_loss_gen'].append(mse_loss_gen.item())
            losses['kld_loss_gen'].append(kld_loss_gen.item())
            losses['wl_loss_gen'].append(wl_loss_gen.item())
            losses['loss_gen'].append(loss_gen.item())

            # Train the Discriminator

            # disc prediction
            disc_pred = discriminator(pred_lab)
            # disc true
            pred_true = torch.cat([trainL, trainAB], dim=1).to(config.DEVICE)
            disc_true = discriminator(pred_true)
            # disc average
            averaged_sample = RandomWeightedAverage([trainAB, predAB])
            average_sample = torch.cat([trainL, averaged_sample], dim=1)
            average_sample = average_sample.to(config.DEVICE)
            average_sample = torch.autograd.Variable(average_sample, requires_grad=True)
            disc_average = discriminator(average_sample)
            mal_F = disc_pred * negative_real
            mal_F = torch.autograd.Variable(mal_F, requires_grad=True)
            mal_R = disc_true * positive_real
            mal_R = torch.autograd.Variable(mal_R, requires_grad=True)
            Loss_D_Fake = wasserstein_loss(mal_F) * -1.0
            Loss_D_Real = wasserstein_loss(mal_R)

            #           print(disc_average.shape, average_sample.shape)
            Loss_D_avg = gradient_penalty_loss(
                disc_average,
                average_sample,
                config.GRADIENT_PENALTY_WEIGHT)
            Loss_D = (Loss_D_Fake + Loss_D_Real + Loss_D_avg)

            # Backpropagation
            optimizer_d.zero_grad()
            Loss_D.backward(retain_graph=True)
            optimizer_d.step()

            # append losses
            losses['Loss_D_Fake'].append(Loss_D_Fake.item())
            losses['Loss_D_Real'].append(Loss_D_Real.item())
            losses['Loss_D_avg'].append(Loss_D_avg.item())
            losses['Loss_D'].append(Loss_D.item())

            # Save losses and images
            if idx % 300 == 0:
                #                 sample_images(test_dataloader, colorization_model, epoch)
                print(f'{idx} / {len(train_dataloader)}')
                print(f'Loss_D_Fake: {Loss_D_Fake.item()}')
                print(f'Loss_D_Real: {Loss_D_Real.item()}')
                print(f'Loss_D_avg: {Loss_D_avg.item()}')
                print(f'Loss_D: {Loss_D.item()}')
                print(f'Loss_gen: {loss_gen.item()}')
                print(f'Loss_mse_gen: {mse_loss_gen.item()}')
                print(f'Loss_kld_gen: {kld_loss_gen.item()}')
                print(f'Loss_wl_gen: {wl_loss_gen.item()}')
                print('-' * 30)

            # sample images after each epoch
    # Save losses
    with open(os.path.join(save_models_path, 'losses.json'), 'w') as f:
        json.dump(losses, f)
    torch.save(discriminator.state_dict(), config.MODEL_PATH + 'discrip.pt')
    torch.save(colorization_model.state_dict(), config.MODEL_PATH + 'color.pt')


def train():
    train_path = config.TRAIN_PATH
    test_path = config.TEST_PATH
    epochs = config.EPORCHS

    print('Start training...')
    print('-' * 30)

    model(train_path, test_path, epochs)

    print('-' * 30)
    print('Training done!')
    print('-' * 30)

    print('Start testing...')
    print('-' * 30)

    test(test_path)

    print('-' * 30)
    print('Testing done!')
    print('-' * 30)


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


class Trainer():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self):
        train()


class Tester():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test(self):
        test()
