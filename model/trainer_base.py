import abc
import numpy as np
import util.vis_util as vis_util

class BaseTrainer():
    def __init__(self, config, dataset, device):
        self.config = config
        self.device = device
        self.dataset = dataset

        self.plot_jnts_fn = self.dataset.plot_jnts if hasattr(self.dataset, 'plot_jnts') and callable(self.dataset.plot_jnts) \
                                                        else vis_util.vis_skel
        self.plot_traj_fn = self.dataset.plot_traj if hasattr(self.dataset, 'plot_traj') and callable(self.dataset.plot_traj) \
                                                        else vis_util.vis_traj
        return

    @abc.abstractmethod
    def train_model(self, model, dataset, epochs):
        for ep in epochs:
            loss_info = self.train_loop(model, dataset)
        return

    @abc.abstractmethod
    def train_loop(self, model):
        return

    @abc.abstractmethod
    def evaluate(self, model, dataset):
        return
    
    

    