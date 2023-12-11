import numpy as np
import abc
import torch

class BaseTask:
    def __init__(self):
        pass

    @abc.abstractmethod
    def set_target(self):
        pass

    @abc.abstractmethod
    def compute_distance(self):
        pass

    @abc.abstractmethod
    def update_state(self):
        pass
    