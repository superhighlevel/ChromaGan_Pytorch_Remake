import numpy as np
import os
import sys
import configs.config_aa as config_aa
import cv2
import torch
import configs.config as config
from functools import partial

def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def reconstruct(batchX, predictedY, filelist):
    """

    """
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)

    save_results_path = os.path.join(config_aa.OUT_DIR,config_aa.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist +  "_reconstructed.jpg" )
    cv2.imwrite(save_path, result)
    return result

def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result

def wasserstein_loss(input, target: list):
    return torch.mean(target)

def RandomWeightedAverage(inputs):
    weight = torch.rand((config.BATCH_SIZE, 1, 1, 1))
    return (weight * inputs[0]) + ((1 - weight) * inputs[1])

def gradient_penalty_loss(
        input, target, 
        averaged_samples, gradient_penalty_weight):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = torch.autograd.grad(outputs=averaged_samples, inputs=input,
                                    grad_outputs=torch.ones(averaged_samples.size()).to(config.DEVICE),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
    return gradient_penalty


def partial_gp_loss(input, target, averaged_samples, gradient_penalty_weight):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    partial_gp_loss = partial(
        gradient_penalty_loss(input, target, averaged_samples, gradient_penalty_weight), 
        averaged_samples)
    return partial_gp_loss





    