import torch
GRADIENT_PENALTY_WEIGHT = 10

EPORCHS = 15

MODEL_PATH = 'models/'
MODEL_NAME =  'ChromaGAN'
MODEL_DIR = 'models/'

TRAIN_PATH = 'datasets/celeb_dataset/train/color'
TEST_PATH = 'datasets/celeb_dataset/test/color'

BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

LR = 0.0002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"