import numpy as np

## Classes and functions for tracking metrics

class EarlyStoppage():

    def __init__(self, trigger_epoch):
        self.trigger_epoch = trigger_epoch
        self.epoch_track = 0
        self.loss_track = np.inf
        self.stop_training = False
        self.improved = False


        return

    def update(self, loss):
        #Always runs first
        self.improved = False
        self.epoch_track += 1

        if loss < self.loss_track: #check if improved
            self.loss_track = loss
            self.epoch_track = 0
            self.improved = True

        if self.epoch_track > self.trigger_epoch: #Check if passed early stoppage
            self.stop_training = True