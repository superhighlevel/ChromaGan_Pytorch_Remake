import os
from .trainer import Trainer

class Train():
    def __init__(self):
        self.trainer = Trainer()
    
    def train(self):
        self.trainer.train()
        print('Training done!')