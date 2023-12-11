import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import util.logging as logging_util
import util.eval as eval_util
import util.vis_util as vis_util
import model.trainer_base as trainer_base

class MVAETrainer(trainer_base.BaseTrainer):
    NAME = 'MVAE'
    def __init__(self, config, dataset, device):
        super(MVAETrainer, self).__init__(config, dataset, device)

        optimizer_config = config['optimizer']
        self.batch_size = optimizer_config['mini_batch_size']
        self.num_rollout = optimizer_config['rollout']
        self.initial_lr = optimizer_config['initial_lr']
        self.final_lr = optimizer_config['final_lr']
        self._get_schedule_samp_routines(optimizer_config)
        
        test_config = config['test']
        self.test_interval = test_config["test_interval"]
        self.test_num_steps = test_config["test_num_steps"]
        self.test_num_trials = test_config["test_num_trials"]
        
        self.frame_dim = dataset.frame_dim
        self.train_dataloader = DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True)
        self.logger = logging_util.wandbLogger()
        self.use_cond = config['model_hyperparam'].get('use_cond', False)

    def _get_schedule_samp_routines(self, optimizer_config):
        self.anneal_times = optimizer_config['anneal_times']
        self.teacher_epochs = optimizer_config['teacher_epochs']
        self.ramping_epochs = optimizer_config['ramping_epochs']
        self.student_epochs = optimizer_config['student_epochs']
        self.use_schedule_samp = self.ramping_epochs != 0 or self.student_epochs != 0
        
        self.sample_schedule = torch.cat([ 
                # First part is pure teacher forcing
                torch.zeros(self.teacher_epochs),
                # Second part with schedule sampling
                torch.linspace(0.0, 1.0, self.ramping_epochs),
                # last part is pure student
                torch.ones(self.student_epochs),
        ])
        self.sample_schedule = torch.cat([self.sample_schedule  for _ in range(self.anneal_times)], axis=-1)
        #self.sample_schedule = torch.cat([self.sample_schedule, torch.zeros(1)])  
        self.total_epochs = self.sample_schedule.shape[0]+1

    def _init_optimizer(self, model):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.initial_lr)

    def _update_lr_schedule(self, optimizer, epoch):
        """Decreases the learning rate linearly"""
        lr = self.initial_lr - (self.initial_lr - self.final_lr) * epoch / float(self.total_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def compute_teacher_loss(self, model, sampled_frames, cur_epoch, extra_info):
        st_index = random.randint(0,sampled_frames.shape[1]-2)
        last_frame = sampled_frames[:,st_index,:]
        ground_truth = sampled_frames[:,st_index+1,:]
        loss, loss_dict = model.compute_loss(last_frame, ground_truth, cur_epoch, extra_info)
        return loss, loss_dict
                    
    def compute_student_loss(self, model, sampled_frames, cur_epoch, extra_info):
        loss_sum = 0
        last_frame = sampled_frames[:,0,:]
        for st_index in range(self.num_rollout - 1):
            next_index = st_index + 1
            ground_truth = sampled_frames[:,next_index,:]
            loss, loss_dict = model.compute_loss(last_frame, ground_truth, cur_epoch, extra_info)

            with torch.no_grad():
                x = model.eval_step(last_frame, extra_info)

            last_frame = x.detach()
            loss_sum += loss 
        return loss_sum
    
    def train_model(self, model, out_model_file, int_output_dir, log_file):
        self._init_optimizer(model)
        for ep in range(0, self.total_epochs+1):
            loss_stats = self.train_loop(ep, model)
            if ep % self.test_interval == 0:
                self.evaluate(model, int_output_dir)
                torch.save(model, int_output_dir+'_ep{}.pt'.format(ep))
            torch.save(model, out_model_file)\

    def train_loop(self, ep, model):
        ep_diff_loss = 0
        num_samples = 0
        self._update_lr_schedule(self.optimizer, ep - 1)        
        model.train()
        loss_dict_sum = {}
        for i, frames in enumerate(self.train_dataloader):
            #skel = self.dataset.x_to_jnts(self.dataset.denorm_data(frames[0].cpu().detach().numpy()))
            #vis_util.vis_skel(skel, self.dataset.links)
            if self.use_cond:
                #### in this case frames = [motion, label_cond]
                frames, cond = frames[0].to(self.device).float(), frames[1].to(self.device).float()
                extra_info = {'cond': cond}
            else:
                frames = frames.to(self.device).float()
                extra_info = None

            if torch.bernoulli(self.sample_schedule[ep]):
                loss, loss_dict = self.compute_student_loss(model, frames, ep, extra_info=extra_info)
                cur_samples = frames.shape[0]
            else:
                loss, loss_dict = self.compute_teacher_loss(model, frames, ep, extra_info=extra_info)
                cur_samples = frames.shape[0] * self.num_rollout
            
            for k in loss_dict:
                if k not in loss_dict_sum:
                    loss_dict_sum[k] = loss_dict[k]
                else:
                    loss_dict_sum[k] += loss_dict[k]

            num_samples += cur_samples
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            ep_diff_loss += loss.item()
        
        for k in loss_dict_sum:
            loss_dict_sum[k] = loss_dict_sum[k] / num_samples
        train_info = {
                    "epoch": ep,
                    "sch_smp_rate": self.sample_schedule[ep],
                    "ep_loss": ep_diff_loss / num_samples,
                    **loss_dict_sum
                }
        
        self.logger.log_epoch(train_info)
        self.logger.print_log(train_info)

    def evaluate(self, model, result_ouput_dir):
        model.eval()
        
        apd_lst = []
        ade_lst = []
        fde_lst = []
        rigid_lst = []
        foot_slide_lst = []
        foot_slide_gt = []

        NaN_clip_num = 0
        for idx, (st_idx, ref_clip) in enumerate(zip(self.dataset.test_valid_idx, self.dataset.test_ref_clips)):
            print('Eval Index:',st_idx)
            test_out_lst = []
            start_x = torch.from_numpy(ref_clip[0]).float().to(self.device)
            if self.use_cond:
                cond = torch.from_numpy(self.dataset.clip_embedding[:self.test_num_trials]).float().to(self.device)
                extra_info = {'cond': cond}
            else:
                extra_info = None
            test_data = model.eval_seq(start_x, extra_info, self.test_num_steps, self.test_num_trials)
            test_data = test_data.detach().cpu().numpy()

            ref_clip = self.dataset.x_to_jnts(self.dataset.denorm_data(ref_clip))
            self.plot_jnts_fn(ref_clip, result_ouput_dir+'/gt_{}'.format(st_idx))    
            ref_clip = ref_clip[None,...]
            should_skip = False
            for i in range(test_data.shape[0]):
                test_clip = self.dataset.x_to_jnts(self.dataset.denorm_data(test_data[i]))
                # skip NaN clips
                try:
                    self.plot_jnts_fn(test_clip, result_ouput_dir+'/{}_{}'.format(st_idx,i))
                    test_out_lst.append(test_clip)
                except:
                    print(idx,i,'NaN')
                    should_skip = True

            if should_skip:
                NaN_clip_num += 1
                continue  # skip stats for this one
            
            if NaN_clip_num >= len(self.dataset.test_valid_idx)-1:
                return # abbadon eval to save time

            test_out_lst = np.array(test_out_lst)
            self.plot_traj_fn(test_out_lst, result_ouput_dir+'/{}'.format(st_idx))
            eval_info = eval_util.compute_test_metrics(self.dataset.links, self.dataset.foot_idx, test_out_lst, ref_clip)
            
            foot_slide_lst.append(eval_info['sliding'])
            foot_slide_gt.append(eval_info['sliding_gt'])
            rigid_lst.append(eval_info['rigid'])
            
            apd_lst.append(eval_info['apd'])
            ade_lst.append(eval_info['ade'])
            fde_lst.append(eval_info['fde'])

        apd_mean, apd_dev = np.mean(apd_lst),np.std(apd_lst)
        ade_mean, ade_dev = np.mean(ade_lst),np.std(ade_lst)
        fde_mean, fde_dev = np.mean(fde_lst),np.std(fde_lst)

        rigid_mean = np.mean(rigid_lst)
        rigid_dev = np.std(rigid_lst)
        slide_mean = np.mean(foot_slide_lst)
        slide_dev = np.std(foot_slide_lst)
        slide_gt = np.mean(foot_slide_gt)

        stats = {
                    'APD': apd_mean,'APD_dev': apd_dev,
                    'ADE': ade_mean,'ADE_dev': ade_dev,
                    'FDE': fde_mean,'FDE_dev': fde_dev,
                    'rigid':rigid_mean,'rigid_dev':rigid_dev,
                    'sliding':slide_mean,'sliding_dev':slide_dev,'sliding_gt': slide_gt
                }
        self.logger.print_log(stats)
        self.logger.log_epoch(stats)