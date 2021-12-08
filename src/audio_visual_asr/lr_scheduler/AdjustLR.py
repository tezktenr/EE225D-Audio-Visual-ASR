"""
Filename: AdjustLR.py
Description: This is a file that contains the class 'AdjustLR', which is a
             scheduler for updating learning rate during learning
"""
# Python Standard Libraries
import math

# Third Party Libraries

# Project Module

# Source Code
class AdjustLR(object):
    """
    This class is responsible for updating the learning rate of the given optimizer during learning
    """

    def __init__(self, optimizer, init_lr, sleep_epochs=5, half=5, verbose=0):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.sleep_epochs = sleep_epochs
        self.half = half
        self.init_lr = init_lr
        self.verbose = verbose

    def step(self, epoch):
        if epoch >= self.sleep_epochs:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.init_lr[idx] * math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
                param_group['lr'] = new_lr
            if self.verbose:
                print('>>> reduce learning rate <<<')