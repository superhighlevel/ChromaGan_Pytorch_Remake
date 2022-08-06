import torch

GRADIENT_PENALTY_WEIGHT = 10

EPOCHS = 2
MODEL_DIR = 'models'
MODEL_C = MODEL_DIR + '/colorization_2.pt'
MODEL_D = MODEL_DIR + '/discriminator.pt'

TRAIN_PATH = 'sample/train'
TEST_PATH = 'sample/test'

OUTPUT_PATH = 'Output/'

BATCH_SIZE = 2
TEST_BATCH_SIZE = 4

# -1 for no log
CHECK_PER = 100

LR = 0.0002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
