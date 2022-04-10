import os
from trainer import Trainer
import configs.config as config


class Train:
    def __init__(self):
        self.trainer = Trainer()

    def train(self):
        self.trainer.train()
        print('Training done!')


# Create output folder if not exist
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
