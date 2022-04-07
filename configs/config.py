import torch

GRADIENT_PENALTY_WEIGHT = 10

EPORCHS = 1

MODEL_PATH = 'models/'
MODEL_NAME = 'ChromaGAN'
MODEL_DIR = 'models/'

OUTPUT_PATH = 'Output/'

# TRAIN_PATH = '../input/celeb-dataset/celeb_dataset/train/color'
TRAIN_PATH = 'datasets/celeb_dataset/test/color'
TEST_PATH = 'datasets/celeb_dataset/test/color'

BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

LR = 0.0002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
