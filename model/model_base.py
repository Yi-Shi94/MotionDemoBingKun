import numpy as np
import abc
import torch

class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        return
        
    @abc.abstractmethod
    def _build_model(self, config):
        return

    @abc.abstractmethod
    def eval_step(self, cur_x, extra_dict):
        return

    @abc.abstractmethod
    def compute_loss(self, cur_x, tar_x, extra_dict):
        return
