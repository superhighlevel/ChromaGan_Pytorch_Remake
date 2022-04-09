import numpy as np
import os
import cv2
import torch
import configs.config as config
from functools import partial
import torch.nn as nn


def deprocess(imgs):
    """
    Deprocess the image
    :param imgs: image to be deprocessed
    :return: deprocessed image
    """
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY):
    """
    Reconstruct the image
    :param batchX: input image
    :param predictedY: predicted image
    :return: reconstructed image
    """
    result = np.concatenate((batchX, predictedY))
    result = np.transpose(result, (1, 2, 0))
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def wasserstein_loss(input):
    """
    Wasserstein loss: https://arxiv.org/abs/1701.07875
    :param input: input to the loss function
    :return: Wasserstein loss
    """
    return torch.mean(input)


def RandomWeightedAverage(inputs):
    """
    Computes a weighted average of the inputs 
    :param inputs: list of input tensors
    :return: a tensor with the same shape as the input tensors
    """
    weight = torch.rand((config.BATCH_SIZE, 1, 1, 1)).to(config.DEVICE)
    return (weight * inputs[0]) + ((1 - weight) * inputs[1])


def gradient_penalty_loss(y_pred, averaged_samples, gradient_penalty_weight):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples: https://arxiv.org/abs/1704.00028
    :param y_pred: prediction
    :param averaged_samples: weighted real / fake samples
    :param gradient_penalty_weight: weight of the gradient penalty
    :return: gradient penalty
    """
    gradients = torch.autograd.grad(y_pred, averaged_samples,
                                    grad_outputs=torch.ones(y_pred.size(), device=config.DEVICE),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
    return gradient_penalty


def partial_gp_loss(target, averaged_samples, gradient_penalty_weight):
    """
    Computes partial gradient penalty loss.
    :param target: target image
    :param averaged_samples: weighted real / fake samples
    :param gradient_penalty_weight: weight of the gradient penalty
    :return: partial gradient penalty
    """
    partial_gp_loss = partial(
        gradient_penalty_loss(
            target, averaged_samples, 
            gradient_penalty_weight),
        averaged_samples)
    return partial_gp_loss


def initialize_weights(model):
    """
    Initialize weights for the model
    :param model: model to be initialized
    :return: initialized model
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
