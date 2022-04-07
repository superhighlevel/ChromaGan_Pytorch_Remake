import numpy as np
import os
import cv2
import torch
import configs.config as config
from functools import partial
import torch.nn as nn


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)

    save_results_path = os.path.join(config.OUT_DIR, config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist + "_reconstructed.jpg")
    cv2.imwrite(save_path, result)
    return result


def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def wasserstein_loss(input):
    return torch.mean(input)


def RandomWeightedAverage(inputs):
    weight = torch.rand((config.BATCH_SIZE, 1, 1, 1)).to(config.DEVICE)
    return (weight * inputs[0]) + ((1 - weight) * inputs[1])


def gradient_penalty_loss(y_pred, averaged_samples, gradient_penalty_weight):
    gradients = torch.autograd.grad(y_pred, averaged_samples,
                                    grad_outputs=torch.ones(y_pred.size(), device=config.DEVICE),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
    return gradient_penalty


def partial_gp_loss(target, averaged_samples, gradient_penalty_weight):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    partial_gp_loss = partial(
        gradient_penalty_loss(target, averaged_samples, gradient_penalty_weight),
        averaged_samples)
    return partial_gp_loss


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
