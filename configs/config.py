import torch

GRADIENT_PENALTY_WEIGHT = 10

EPOCHS = 10
MODEL_DIR = 'models'
MODEL_C = MODEL_DIR + '/colorization.pt'
MODEL_D = MODEL_DIR + '/discriminator.pt'

TRAIN_PATH = 'sample/train'
TEST_PATH = 'sample/test'

OUTPUT_PATH = 'Output/'

BATCH_SIZE = 2
TEST_BATCH_SIZE = 4

# -1 for no log
CHECK_PER = 100

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
