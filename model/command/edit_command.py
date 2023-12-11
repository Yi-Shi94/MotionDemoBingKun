import numpy as np
import model.command.base_command as base_command
import torch

class Edit(base_command.BaseCommandTask):
    NAME = "Edit"
    def __init__(self, test_config):
        super(Edit).__init__()
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
        self.init_frame = torch.tensor(self.init_frame)[0].to(self.device).float()
        self.test_out = int_output_dir

        self.heading_signal = None
        self.jointpos_signal = None
        self.jointangle_signal = None
        self.rootdxdy_signal = None
        
        self.prepare_edit_control()

    def parse_signal(self, signal):
        if isinstance(signal, str):
            return torch.tensor(np.load(signal))
            #assert signal.shape[0] ==  
        elif isinstance(signal, float):
            return signal

        elif isinstance(signal, int):
            return signal*1.0

    def prepare_edit_control(self):
        edited_data = torch.zeros(1,init_frame).expand(self.test_num_steps, -1).float()
        edited_label = torch.zeros_like(edited_data)
        config = self.config["edit"]

        if "joint_pos" in config:
            joint_config = config["joint_pos"]
            for key in joint_config.item():
                edit_dim = self.dataset.get_dim_by_key("joint_pos", key)  #todo
                if edit_dim is None:
                    continue    
                #if self.jointpos_signal is None:
                #    self.jointpos_signal = {}
                edited_data[...,edit_dim] = self.parse_signal(joint_config[key]) 
                edited_label[...,edit_dim] = 1

        if "joint_angle" in config:
            angle_config = config["joint_angle"]
            for key in angle_config.item():
                edit_dim = self.dataset.get_dim_by_key("joint_angle", key) #todo
                if edit_dim is None:
                    continue    
                edited_data[...,edit_dim] = self.parse_signal(joint_config[key])
                edited_label[...,edit_dim] = 1

        if "joint_vel" in config:
            vel_config = config["joint_vel"]
            for key in vel_config.item():
                edit_dim = self.dataset.get_dim_by_key("joint_vel", key) #todo
                if edit_dim is None:
                    continue    
                edited_data[...,edit_dim] = self.parse_signal(joint_config[key])
                edited_label[...,edit_dim] = 1
        
        if "heading" in config:
            heading = config["heading"]
            edit_dim = self.dataset.get_dim_by_key("heading",None)
            edited_data[...,edit_dim] = self.parse_signal(heading)
            edited_label[...,edit_dim] = 1
        
        if "rootdxdy" in config:
            rootdxdy = config["rootdxdy"]
            edit_dim = self.dataset.get_dim_by_key("rootdxdy",None)
            edited_data[...,edit_dim] = self.parse_signal(rootdxdy)
            edited_label[...,edit_dim] = 1

        self.edit_data = self.dataset.norm_data(edited_data)
    
    def play(self):
        seqs = self.model.eval_seq(self.init_frame, self.edit_data, self.test_num_steps, self.test_num_trials)
        for i in range(self.test_num_trials):
            denormed_data = self.dataset.denorm_data(seqs[i])
            joints= self.dataset.x_to_jnts(denormed_data)
            self.trainer.plot_jnts_fn(joints, self.test_out+'/{}'.format(i))

    