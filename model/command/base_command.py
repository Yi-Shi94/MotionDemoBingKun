
import numpy as np
import abc
import torch

class BaseCommandTask:
    def __init__(self):
        pass
        #self.model = model
        #self.dataset = dataset

    @abc.abstractmethod
    def insert_command(self):
        raise NotImplementedError

    @abc.abstractmethod
    def playout(self):
        raise NotImplementedError
