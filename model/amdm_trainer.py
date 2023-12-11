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


class AMDMTrainer(trainer_base.BaseTrainer):
    NAME = 'AMDM'
    def __init__(self, config, dataset, device):
        super(AMDMTrainer, self).__init__(config, dataset, device)

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

    def train_model(self, model, out_model_file, int_output_dir, log_file):
        #model_name = os.path.basename(out_model_file).split('.')[0]
        #self.wanb_run_name = self.logger.run_name if self.logger.run_name else model_name
        self._init_optimizer(model)
        for ep in range(0, self.total_epochs+1):
            loss_stats = self.train_loop(ep, model)
            if ep % self.test_interval == 0:
                eval_stats = self.evaluate(model, int_output_dir)
                torch.save(model, int_output_dir+'_ep{}.pt'.format(ep))
            for name_metric in eval_stats:
                loss_stats[name_metric] = eval_stats[name_metric]
            
            self.logger.log_epoch(loss_stats)
            self.logger.print_log(loss_stats)
            
            torch.save(model, out_model_file)

    def compute_teacher_loss(self, model, sampled_frames, extra_info):
        st_index = random.randint(0,sampled_frames.shape[1]-2)
        last_frame = sampled_frames[:,st_index,:]
        ground_truth = sampled_frames[:,st_index+1,:]
        loss_diff = model.compute_loss(last_frame,  ground_truth,  extra_info)
        return loss_diff
                    
    def compute_student_loss(self, model, sampled_frames, extra_info):
        loss = 0
        last_frame = sampled_frames[:,0,:]
        for st_index in range(self.num_rollout - 1):
            next_index = st_index + 1

            ground_truth = sampled_frames[:,next_index,:]
            loss_diff = model.compute_loss(last_frame, ground_truth, extra_info)
            
            with torch.no_grad():
                x = model.eval_step(last_frame, extra_info)

            last_frame = x.detach()
            loss +=  loss_diff 
        return loss_diff
    

    def train_loop(self, ep, model):
        ep_diff_loss = 0
        num_samples = 0
        self._update_lr_schedule(self.optimizer, ep - 1)
        
        model.train()
        for i, frames in enumerate(self.train_dataloader):
            #skel = self.dataset.x_to_jnts(self.dataset.denorm_data(frames[0].cpu().detach().numpy()))
            #vis_util.vis_skel(skel, self.dataset.links)
            frames = frames.to(self.device).float()
            extra_info = None

            if torch.bernoulli(self.sample_schedule[ep]):
                loss_diff = self.compute_student_loss(model, frames, extra_info=extra_info)
                for g in self.optimizer.param_groups:
                    g['lr'] = self.final_lr
                cur_samples = 1
            else:
                loss_diff = self.compute_teacher_loss(model, frames, extra_info=extra_info)
                cur_samples = 1 * self.num_rollout
            
            num_samples += cur_samples
            self.optimizer.zero_grad()
            loss_diff.backward()
            self.optimizer.step()
            
            model.update()

            ep_diff_loss += loss_diff.item()

        train_info = {
                    "epoch": ep,
                    "sch_smp_rate": self.sample_schedule[ep],
                    "ep_diff_loss": ep_diff_loss / num_samples,
                }
        
        return train_info

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

        eval_info = {
                    'APD': apd_mean,'APD_dev': apd_dev,
                    'ADE': ade_mean,'ADE_dev': ade_dev,
                    'FDE': fde_mean,'FDE_dev': fde_dev,
                    'rigid':rigid_mean,'rigid_dev':rigid_dev,
                    'sliding':slide_mean,'sliding_dev':slide_dev,'sliding_gt': slide_gt
                }
        return eval_info