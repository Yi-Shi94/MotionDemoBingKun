import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.model_base as model_base

class HUMOR(model_base.BaseModel):
    NAME = 'HUMOR'
    def __init__(self, config, dataset, device):
        super().__init__(config)
        self.config = config
        self.device = device
        self.frame_dim = dataset.frame_dim
        
        config['frame_dim'] = self.frame_dim
        self._build_model(config)
        return

    def _build_model(self, config):
        model_name = config["model_name"]
        self.model = HumorModel(config)
        self.loss_fn = HumorLoss()
        self.model.to(self.device)
        return

    def eval_step(self, cur_x, extra_dict):   
        next_x = self.model.sample(cur_x) 
        return next_x

    def eval_seq(self, start_x, extra_dict, num_steps, num_trials):
        output_xs = torch.zeros((num_trials, num_steps, self.frame_dim))
        start_x = start_x[None,:].expand(num_trials, -1)
        for j in range(num_steps):
            with torch.no_grad():
                start_x = self.eval_step(start_x, extra_dict).detach()
            output_xs[:,j,:] = start_x 
        return output_xs
    
    def compute_loss(self, last_x, next_x, cur_epoch, extra_dict):
        decoder_out, posterior, prior = self.model.train_step(last_x, next_x)
        pred_dict = {'pred':decoder_out, 'posterior_distrib':posterior, 'prior_distrib':prior} 
        loss, loss_dict = self.loss_fn(pred_dict, next_x, cur_epoch)
        return loss, loss_dict

    def get_model_params(self):
        params = list(self.model.parameters())
        return params
    

class HumorModel(nn.Module):
    def __init__(self,  config):
        super(HumorModel, self).__init__()
        
        self.ignore_keys = []
        self.latent_size = config["model_hyperparam"]["hidden_size"]
        self.layer_num = config["model_hyperparam"]["layer_num"]
        self.frame_size = config['frame_dim']

        past_data_dim = self.frame_size 
        t_data_dim = self.frame_size 

        self.decoder_arch = 'mlp'
        self.posterior_arch = 'mlp'
        self.prior_arch  = 'mlp'
        
        self.use_conditional_prior = True
        # posterior encoder (given past and future, predict latent transition distribution)
        print('Using posterior architecture: %s' % (self.posterior_arch))
        layer_list = [past_data_dim + t_data_dim, 1024, 1024, 1024, 1024, self.latent_size*2]
        self.encoder = MLP(layers=layer_list, # mu and sigma output
                            nonlinearity=nn.ReLU,
                            use_gn=True
                        )

        # decoder (given past and latent transition, predict future) for the immediate next step
        print('Using decoder architecture: %s' % (self.decoder_arch))
        decoder_input_dim = past_data_dim + self.latent_size
        layer_list = [decoder_input_dim, 1024, 1024, 512, self.frame_size]
        self.decoder = MLP(layers=layer_list,
                            nonlinearity=nn.ReLU, 
                            use_gn=True,
                            skip_input_idx=past_data_dim # skip connect the latent to every layer
                        )

        # prior (if conditional, given past predict latent transition distribution)
        print('Using prior architecture: %s' % (self.prior_arch))
        layer_list = [past_data_dim, 1024, 1024, 1024, 1024, self.latent_size*2]
        self.prior_net = MLP(layers=layer_list, # mu and sigma output
                        nonlinearity=nn.ReLU, 
                        use_gn=True
                    )
    

    def forward(self, x_past, x_t):
        '''
        single step full forward pass. This uses the posterior for sampling, not the prior.
        
        Input:
        - x_past (B x D)
        - x_t    (B x D)

        Returns dict of:
        - x_pred (B x D)
        - posterior_distrib (Normal(mu, sigma))
        - prior_distrib (Normal(mu, sigma))
        '''

        B, D = x_past.size()
        past_in = x_past.reshape((B, -1))
        t_in = x_t.reshape((B, -1))
        output, dist_p, dist_q = self.single_step(past_in, t_in)
        return output, dist_p, dist_q


    def train_step(self, past_in, t_in):
        '''
        single step that computes both prior and posterior for training. Samples from posterior
        '''
        B = past_in.size(0)
        # use past and future to encode latent transition
        qm, qv = self.posterior(past_in, t_in)

        # prior
        pm, pv = None, None
        if self.use_conditional_prior:
            # predict prior based on past
            pm, pv = self.prior(past_in)
        else:
            # use standard normal
            pm, pv = torch.zeros_like(qm), torch.ones_like(qv)

        # sample from posterior using reparam trick
        z = self.rsample(qm, qv)

        # decode to get next step
        decoder_out = self.decode(z, past_in)
        decoder_out = decoder_out.reshape((B, -1)) # B  x D_out
        return decoder_out, (pm, pv), (qm, qv)


    def prior(self, past_in):
        '''
        Encodes the posterior distribution using the past and future states.

        Input:
        - past_in (B x steps_in*D)
        '''
        prior_out = self.prior_net(past_in)
        mean = prior_out[:,:self.latent_size]
        logvar = prior_out[:,self.latent_size:]
        var = torch.exp(logvar)
        return mean, var

    def posterior(self, past_in, t_in):
        '''
        Encodes the posterior distribution using the past and future states.

        Input:
        - past_in (B x steps_in*D)
        - t_in    (B x steps_out*D)
        '''
        encoder_in = torch.cat([past_in, t_in], axis=1)

        encoder_out = self.encoder(encoder_in)
        mean = encoder_out[:,:self.latent_size]
        logvar = encoder_out[:,self.latent_size:]
        var = torch.exp(logvar)

        return mean, var


    def rsample(self, mu, var):
        '''
        Return gaussian sample of (mu, var) using reparameterization trick.
        '''
        eps = torch.randn_like(mu)
        z = mu + eps*torch.sqrt(var)
        return z


    def decode(self, z, past_in):
        '''
        Decodes prediction from the latent transition and past states

        Input:
        - z  (B x latent_size)
        - past_in (B x D)

        Returns:
        - decoder_out (B x D)
        '''
        B = z.size(0)
        decoder_in = torch.cat([past_in, z], axis=1)
        decoder_out = self.decoder(decoder_in).reshape((B, 1, -1))
        
        decoder_out = decoder_out.reshape((B, -1))
        return decoder_out


    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x D
        '''
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((pad_size, pad_tensor.size(1))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
        return new_pad_list

            
    def sample(self, past_in, t_in=None, use_mean=False, z=None, return_prior=False, return_z=False):
        '''
        Given past, samples next future state by sampling from prior or posterior and decoding.
        If z (B x D) is not None, uses the given z instead of sampling from posterior or prior

        Returns:
        - decoder_out : (B x steps_out x D) output of the decoder for the immediate next step
        '''
        B = past_in.size(0)

        pm, pv = None, None
        if t_in is not None:
            # use past and future to encode latent transition
            pm, pv = self.posterior(past_in, t_in)
        else:
            # prior
            if self.use_conditional_prior:
                # predict prior based on past
                pm, pv = self.prior(past_in)
            else:
                # use standard normal
                pm, pv = torch.zeros((B, self.latent_size)).to(past_in), torch.ones((B, self.latent_size)).to(past_in)

        # sample from distrib or use mean
        if z is None:
            if not use_mean:
                z = self.rsample(pm, pv)
            else:
                z = pm # NOTE: use mean

        # decode to get next step
        decoder_out = self.decode(z, past_in)
        decoder_out = decoder_out.reshape((B, -1)) 

        out_dict = {'decoder_out' : decoder_out}
        if return_prior:
            out_dict['prior'] = (pm, pv)
        if return_z:
            out_dict['z'] = z
        
        return out_dict['decoder_out']


    def infer(self, x_past, x_t):
        '''
        Inference (compute prior and posterior distribution of z) for a batch of single steps.
        NOTE: must do processing before passing in to ensure correct format that this function expects.
        
        Input:
        - x_past (B x steps_in x D)
        - x_t    (B x steps_out x D)

        Returns:
        - prior_distrib (mu, var)
        - posterior_distrib (mu, var)
        '''

        B, D = x_past.size()
        past_in = x_past.reshape((B, -1))
        t_in = x_t.reshape((B, -1))
        prior_z, posterior_z = self.infer_step(past_in, t_in)
        return prior_z, posterior_z



    def infer_step(self, past_in, t_in):
        '''
        single step that computes both prior and posterior for training. Samples from posterior
        '''
        B = past_in.size(0)
        # use past and future to encode latent transition
        qm, qv = self.posterior(past_in, t_in)

        # prior
        pm, pv = None, None
        if self.use_conditional_prior:
            # predict prior based on past
            pm, pv = self.prior(past_in)
        else:
            # use standard normal
            pm, pv = torch.zeros_like(qm), torch.ones_like(qv)

        return (pm, pv), (qm, qv)
    


def step(model, loss_func, data, device, cur_epoch, use_gt_p=1.0):
    '''
    Given data for the current training step (batch),
    pulls out the necessary needed data,
    runs the model,
    calculates and returns the loss.

    - use_gt_p : the probability of using ground truth as input to each step rather than the model's own prediction
                 (1.0 is fully supervised, 0.0 is fully autoregressive)
    '''
    use_sched_samp = use_gt_p < 1.0
    batch_in, batch_out, meta = data

    prep_data = model.prepare_input(batch_in, device, data_out=batch_out, return_input_dict=True, return_global_dict=use_sched_samp)
    if use_sched_samp:
        x_past, x_t, gt_dict, input_dict, global_gt_dict = prep_data
    else:
        x_past, x_t, gt_dict, input_dict = prep_data

    B, T, S_in, _ = x_past.size()
    S_out = x_t.size(2)

    if not use_sched_samp:
        # fully supervised phase
        # start by using gt at every step, so just form all steps from all sequences into one large batch
        #       and get per-step predictions
        x_past_batched = x_past.reshape((B*T, S_in, -1))
        x_t_batched = x_t.reshape((B*T, S_out, -1))
        out_dict = model(x_past_batched, x_t_batched)
    else:
        # in scheduled sampling or fully autoregressive phase
        init_input_dict = dict()
        for k in input_dict.keys():
            init_input_dict[k] = input_dict[k][:,0,:,:] # only need first step for init
        # this out_dict is the global state
        sched_samp_out = model.scheduled_sampling(x_past, x_t, init_input_dict, p=use_gt_p, 
                                                                    gender=meta['gender'],
                                                                    betas=meta['betas'].to(device),
                                                                    need_global_out=(not model.detach_sched_samp))
        if model.detach_sched_samp:
            out_dict = sched_samp_out
        else:
            out_dict, _ = sched_samp_out
        # gt must be global state for supervision in this case
        if not model.detach_sched_samp:
            print('USING global supervision')
            gt_dict = global_gt_dict

    # loss can be computed per output step in parallel
    # batch dicts accordingly
    for k in out_dict.keys():
        if k == 'posterior_distrib' or k == 'prior_distrib':
            m, v = out_dict[k]
            m = m.reshape((B*T, -1))
            v = v.reshape((B*T, -1))
            out_dict[k] = (m, v)
        else:
            out_dict[k] = out_dict[k].reshape((B*T*S_out, -1))
    for k in gt_dict.keys():
        gt_dict[k] = gt_dict[k].reshape((B*T*S_out, -1))

    gender_in = np.broadcast_to(np.array(meta['gender']).reshape((B, 1, 1, 1)), (B, T, S_out, 1))
    gender_in = gender_in.reshape((B*T*S_out, 1))
    betas_in = meta['betas'].reshape((B, T, 1, -1)).expand((B, T, S_out, 16)).to(device)
    betas_in = betas_in.reshape((B*T*S_out, 16))
    loss, stats_dict = loss_func(out_dict, gt_dict, cur_epoch, gender=gender_in, betas=betas_in)

    return loss, stats_dict


class MLP(nn.Module):
    def __init__(self, layers=[3, 128, 128, 3], nonlinearity=nn.ReLU, use_gn=True, skip_input_idx=None):
        '''
        If skip_input_idx is not None, the input feature after idx skip_input_idx will be skip connected to every later of the MLP.
        '''
        super(MLP, self).__init__()

        in_size = layers[0]
        out_channels = layers[1:]

        # input layer
        layers = []
        layers.append(nn.Linear(in_size, out_channels[0]))
        skip_size = 0 if skip_input_idx is None else (in_size - skip_input_idx)
        # now the rest
        for layer_idx in range(1, len(out_channels)):
            fc_layer = nn.Linear(out_channels[layer_idx-1] + skip_size, out_channels[layer_idx])
            if use_gn:
                bn_layer = nn.GroupNorm(16, out_channels[layer_idx-1])
                layers.append(bn_layer)
            layers.extend([nonlinearity(), fc_layer])
        self.net = nn.ModuleList(layers)
        self.skip_input_idx = skip_input_idx

    def forward(self, x):
        '''
        B x D x * : batch norm done over dim D
        '''
        skip_in = None
        if self.skip_input_idx is not None:
            skip_in = x[:,self.skip_input_idx:]
        for i, layer in enumerate(self.net):
            if self.skip_input_idx is not None and i > 0 and isinstance(layer, nn.Linear):
                x = torch.cat([x, skip_in], dim=1)
            x = layer(x)
        return x


class HumorLoss(nn.Module):
    def __init__(self,
                    kl_loss=0.0004,
                    kl_loss_anneal_start=0,
                    kl_loss_anneal_end=50,
                    kl_loss_cycle_len=-1, # if > 0 will anneal KL loss cyclicly
                    regr_trans_loss=1.0,
                    regr_trans_vel_loss=1.0,
                    regr_root_orient_loss=1.0,
                    regr_root_orient_vel_loss=1.0,
                    regr_pose_loss=1.0,
                    regr_pose_vel_loss=1.0,
                    regr_joint_loss=1.0,
                    regr_joint_vel_loss=1.0,
                    regr_joint_orient_vel_loss=1.0,
                    regr_vert_loss=1.0,
                    regr_vert_vel_loss=1.0,
                    contacts_loss=0.0, # classification loss on binary contact prediction
                    contacts_vel_loss=0.0, # velocity near 0 at predicted contacts
                    ):
        super(HumorLoss, self).__init__()
        '''
        All loss inputs are weights for that loss term. If the weight is 0, the loss is not used.

        - regr_*_loss :                 L2 regression losses on various state terms (root trans/orient, body pose, joint positions, and joint velocities)
        - smpl_joint_loss :             L2 between GT joints and joint locations resulting from SMPL model (parameterized by trans/orient/body poase)
        - smpl_mesh_loss :              L2 between GT and predicted vertex locations resulting from SMPL model (parameterized by trans/orient/body poase)
        - smpl_joint_consistency_loss : L2 between regressed joints and predicted joint locations from SMPL model (ensures consistency between
                                        state joint locations and joint angle predictions)
        - kl_loss :                     divergence between predicted posterior and prior

        - smpl_batch_size : the size of batches that will be given to smpl. if less than this is passed in, will be padded accordingly. however, passed
                            in batches CANNOT be greater than this number.
        '''
        self.kl_loss_weight = kl_loss
        self.kl_loss_anneal_start = kl_loss_anneal_start
        self.kl_loss_anneal_end = kl_loss_anneal_end
        self.use_kl_anneal = self.kl_loss_anneal_end > self.kl_loss_anneal_start

        self.regr_loss_weight = regr_trans_loss
        self.kl_loss_cycle_len = kl_loss_cycle_len
        self.use_kl_cycle = False
        if self.kl_loss_cycle_len > 0:
            self.use_kl_cycle = True
            self.use_kl_anneal = False

        self.contacts_loss_weight = contacts_loss
        self.contacts_vel_loss_weight = contacts_vel_loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # build dict of all possible regression losses based on inputs
        # keys must be the same as we expect from the pred/gt data
        self.regr_loss_weight_dict = {
            'trans' : regr_trans_loss,
            'trans_vel' : regr_trans_vel_loss,
            'root_orient' : regr_root_orient_loss,
            'root_orient_vel' : regr_root_orient_vel_loss,
            'pose_body' : regr_pose_loss,
            'pose_body_vel' : regr_pose_vel_loss,
            'joints' : regr_joint_loss,
            'joints_vel' : regr_joint_vel_loss,
            'joints_orient_vel' : regr_joint_orient_vel_loss,
            'verts' : regr_vert_loss,
            'verts_vel' : regr_vert_vel_loss
        }

        self.l2_loss = nn.MSELoss(reduction='none')
        #self.regr_loss = nn.MSELoss(reduction='none')


    def forward(self, pred_dict, gt, cur_epoch):
        '''
        Compute the loss.

        All data in the dictionaries should be of size B x D.

        group_regr_losses will be used to aggregate every group_regr_lossth batch idx together into
        the stats_dict. This can be useful when there are multiple output steps and you want to track
        each separately.
        '''
        loss = 0.0
        stats_dict = dict()

        if self.kl_loss_weight > 0.0:
            qm, qv = pred_dict['posterior_distrib']
            pm, pv = pred_dict['prior_distrib']
            kl_loss = self.kl_normal(qm, qv, pm, pv)
            kl_stat_loss = kl_loss.mean()
            kl_loss = kl_stat_loss
            stats_dict['kl_loss'] = kl_stat_loss
            anneal_weight = 1.0
            if self.use_kl_anneal or self.use_kl_cycle:
                anneal_epoch = cur_epoch
                anneal_start = self.kl_loss_anneal_start
                anneal_end = self.kl_loss_anneal_end
                if self.use_kl_cycle:
                    anneal_epoch = cur_epoch % self.kl_loss_cycle_len
                    anneal_start = 0
                    anneal_end = self.kl_loss_cycle_len // 2 # optimize full weight for second half of cycle
                if anneal_epoch >= anneal_start:
                    anneal_weight = (anneal_epoch - anneal_start) / (anneal_end - anneal_start)
                else:
                    anneal_weight = 0.0
                anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight

            loss = loss + anneal_weight*self.kl_loss_weight*kl_loss

            stats_dict['kl_anneal_weight'] = anneal_weight
            stats_dict['kl_weighted_loss'] = loss

        # Reconstruction 
        # regression terms
        
        regr_loss = torch.mean(self.l2_loss(pred_dict['pred'], gt))
        stats_dict['reconstr_weighted_loss'] = regr_loss
        loss = loss + self.regr_loss_weight * regr_loss   
        return loss, stats_dict


    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x D
        '''
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((pad_size, pad_tensor.size(1))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
        return new_pad_list

    
    def kl_normal(self, qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension
        ​
        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance
        ​
        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def log_normal(self, x, m, v):
        """
        Computes the elem-wise log probability of a Gaussian and then sum over the
        last dim. Basically we're assuming all dims are batch dims except for the
        last dim.    Args:
            x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
            m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
            v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance    Return:
            log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
                each sample. Note that the summation dimension is not kept
        """
        log_prob = -torch.log(torch.sqrt(v)) - math.log(math.sqrt(2*math.pi)) \
                        - ((x - m)**2 / (2*v))
        log_prob = torch.sum(log_prob, dim=-1)
        return log_prob
