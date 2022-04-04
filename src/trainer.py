import torch
from src.discriminator import Discriminator
from src.colorization import Colorization
from configs.config_aa import config
from utils.utils import deprocess, reconstruct, reconstruct_no
from torch.utils.data import DataLoader
from src.ColorizeDataloader import ColorizeDataLoader
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.models as models
from utils.utils import *
from torch.optim import adam
import json

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
    train_dataloader = DataLoader(
        train_dataloader, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=4)

    test_dataloader = ColorizeDataLoader(test_data)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=4)

    # Load the discriminator model and the colorization model   
    discriminator = Discriminator(input_size=224, in_channels=3).to(config.DEVICE)
    colorizationModel = Colorization(input_size=224, in_channels=3).to(config.DEVICE)
    VGG_modelF = models.vgg16(pretrained=True)

    # Real, Fake and Dummy for Discriminator
    positive_real = torch.ones(size = (config.BATCH_SIZE, 1))
    negative_real = -positive_real
    dummy_y = torch.zeros(size = (config.BATCH_SIZE, 1))

    optimizer_g = adam(
        colorizationModel.parameters(), lr=config.LR, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer_d = adam(
        discriminator.parameters(), lr=config.LR, 
        beta=(0.9, 0.999), eps=1e-08, weight_decay=0)

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
        print('-'*30)
        for idx, (trainL, trainAB, _, _, _) in enumerate(tqdm(train_dataloader)):
            l_3 = np.title(trainL, [1, 1, 1, 3])

            # Train the Generator
            predictVGG = VGG_modelF.predict(l_3)
            optimizer_g.zero_grad()
            
            predAB, pred_class = colorizationModel(trainL)
            predAB = predAB.detach()
            pred_class = pred_class.detach()
            disc_pred = discriminator(predAB)

            # Loss genrenator 
            mse_loss_gen = torch.nn.functional.mse_loss(predAB, trainL)
            kld_loss_gen = torch.nn.functional.kld_loss(pred_class, predictVGG.detach()) * 0.003
            wl_loss_gen = wasserstein_loss(disc_pred, positive_real) * -0.1
            loss_gen = mse_loss_gen + kld_loss_gen + wl_loss_gen

            # Backpropagation
            color_scaler.scale(loss_gen).backward()
            color_scaler.step(optimizer_g)
            color_scaler.update()

            # append losses
            losses['mse_loss_gen'].append(mse_loss_gen.item())
            losses['kld_loss_gen'].append(kld_loss_gen.item())
            losses['wl_loss_gen'].append(wl_loss_gen.item())
            losses['loss_gen'].append(loss_gen.item())

            

            # Train the Discriminator
            optimizer_d.zero_grad()

            # disc prediction
            pred_lab = torch.cat((trainL, predAB), dim=1)
            disc_pred = discriminator(pred_lab.detach())

            # disc true
            pred_true = torch.cat([trainL, trainAB], dim=1)
            disc_true = discriminator(pred_true)
            
            # disc average
            average_sample = RandomWeightedAverage([trainAB, predAB])
            average_sample = average_sample.detach()
            disc_average = discriminator(average_sample)

            Loss_D_Fake = wasserstein_loss(disc_pred, negative_real) * -1.0
            Loss_D_Real = wasserstein_loss(disc_true, positive_real) 
            Loss_D_avg = gradient_penalty_loss(disc_average, average_sample, config.GRADIENT_PENALTY_WEIGHT)
            Loss_D = Loss_D_Fake + Loss_D_Real + Loss_D_avg

            # Backpropagation
            disc_scaler.scale(Loss_D).backward()
            disc_scaler.step(optimizer_d)
            disc_scaler.update()

            # append losses
            losses['Loss_D_Fake'].append(Loss_D_Fake.item())
            losses['Loss_D_Real'].append(Loss_D_Real.item())
            losses['Loss_D_avg'].append(Loss_D_avg.item())
            losses['Loss_D'].append(Loss_D.item())


            # Save losses and images
            if idx % config.SAVE_FREQ == 0:
                print(f'{idx} / {len(train_dataloader)}')
                print(f'Loss_D_Fake: {Loss_D_Fake.item()}')
                print(f'Loss_D_Real: {Loss_D_Real.item()}')
                print(f'Loss_D_avg: {Loss_D_avg.item()}')
                print(f'Loss_D: {Loss_D.item()}')
                print(f'Loss_gen: {loss_gen.item()}')
                print(f'Loss_mse_gen: {mse_loss_gen.item()}')
                print(f'Loss_kld_gen: {kld_loss_gen.item()}')
                print(f'Loss_wl_gen: {wl_loss_gen.item()}')
                print('-'*30)

            # sample images after each epoch
        # sample_images(test_data, config.EPORCHS)
    # Save losses
    with open(os.path.join(save_models_path, 'losses.json'), 'w') as f:
        json.dump(losses, f)
    

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

def test_case_1(device):
    print('test case 1 start')
    positive_real = torch.ones(size = (config.BATCH_SIZE, 1))
    negative_real = -positive_real
    dummy_y = torch.zeros(size = (config.BATCH_SIZE, 1))
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

    
