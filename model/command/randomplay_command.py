import numpy as np
import model.command.base_command as base_command
import torch

class RandomPlay(base_command.BaseCommandTask):
    NAME = "randomplay"
    def __init__(self, test_config):
        super(RandomPlay).__init__()
        self.config = test_config
        self.test_num_steps = test_config["test_num_steps"]
        self.test_num_trials = test_config["test_num_trials"]
        self.test_init_frame = test_config["test_num_init_frame"]
        
    def prepare(self, model, trainer, int_output_dir, device):
        self.model = model
        self.device = device
        self.trainer = trainer
        self.dataset = trainer.dataset
        self.extra_dict = None
        self.init_frame = self.dataset[self.test_init_frame]
        self.init_frame = torch.tensor(self.init_frame)[0].float().to(self.device)
        self.test_out = int_output_dir

    def play(self):
        seqs = self.model.eval_seq(self.init_frame, self.extra_dict, self.test_num_steps, self.test_num_trials).to('cpu').detach().numpy()
        for i in range(self.test_num_trials):
            denormed_data = self.dataset.denorm_data(seqs[i])
            joints= self.dataset.x_to_jnts(denormed_data)
            self.trainer.plot_jnts_fn(joints, self.test_out+'/{}'.format(i))

    

