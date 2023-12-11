import numpy as np
import model.command.base_command as base_command
import torch
import tqdm

class RandomPlayCLIP(base_command.BaseCommandTask):
    NAME = "randomplay_clip"
    def __init__(self, test_config):
        super(RandomPlayCLIP).__init__()
        self.config = test_config
        self.test_num_steps = test_config["test_num_steps"]
        self.test_num_trials = test_config["test_num_trials"]
        self.test_init_frame = test_config["test_num_init_frame"]
        self.test_texts = test_config["texts"]

    def prepare(self, model, trainer, int_output_dir, device):
        self.model = model
        self.device = device
        self.trainer = trainer
        self.dataset = trainer.dataset

        self.text_embs = self.dataset.get_clip_class_embedding(self.test_texts, outformat='pt')
        self.cond_lst = [{'cond':emb.to(self.device)} for emb in self.text_embs]
        
        self.init_frame, self.init_cond = self.dataset[self.test_init_frame]
        self.init_frame = torch.tensor(self.init_frame)[0].float().to(self.device)
        self.init_cond = torch.tensor(self.init_cond)[0].float().to(self.device)
        
        self.test_out = int_output_dir

    def play(self):
        for j,extra_dict in tqdm.tqdm(enumerate(self.cond_lst)):
            try:
                seqs = self.model.eval_seq(self.init_frame, extra_dict, self.test_num_steps, self.test_num_trials).to('cpu').detach().numpy()
                for i in range(self.test_num_trials):
                    denormed_data = self.dataset.denorm_data(seqs[i])
                    joints= self.dataset.x_to_jnts(denormed_data)
                    self.trainer.plot_jnts_fn(joints, self.test_out+'/{}-{}'.format(self.test_texts[j],i))
            except:
                print('NaN')

    

