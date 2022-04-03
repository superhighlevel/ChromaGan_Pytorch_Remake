import os
from .trainer import Trainer



class Train():
    def __init__(self, config):
        self.config = config
        self.trainer = Trainer(self.config)
    
    def train(self):
        self.trainer.train()
        print('Training done!')
    
    def test(self):
        self.test
        print('Testing done!')
