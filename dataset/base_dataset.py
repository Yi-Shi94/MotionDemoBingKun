import os.path as osp
import numpy as np

from dataset.util.skeleton_info import skel_dict
import dataset.util.file as file_util
import torch.utils.data as data
import tqdm

class BaseMotionData(data.Dataset):
    # For a directory contains multiple identical file type
    def __init__(self, config):
        self.dataset_name = config["data"]["dataset_name"]

        self.skel_info = skel_dict[self.dataset_name]
        self.links = self.skel_info['links']
        
        self.num_jnt = self.skel_info['num_joint']
        self.root_idx = self.skel_info['root_idx']
        self.foot_idx= self.skel_info['foot_idx']
        self.unit = self.skel_info['unit']
        self.fps = self.skel_info['fps']

        self.path = config["data"]["path"]
        self.min_motion_len = config["data"]["min_motion_len"]
        self.max_motion_len = config["data"]["max_motion_len"]
        self.rollout = config["optimizer"]["rollout"]
        self.use_cond = config["model_hyperparam"]["use_cond"]
        
        self.test_num_init_frame = config["test"]["test_num_init_frame"]
        self.test_num_steps = config["test"]["test_num_steps"]
        self.data_trim_begin = config["data"]["data_trim_begin"]
        self.data_trim_end = config["data"]["data_trim_end"]

        self.num_file = 0
        self.file_lst = []
        
        self.extra = dict() # for labels and other multi-modal data (text & audio & video)
        self.labels = list()

        self.motion_lst = list()
        self.label_lst = list() # for labels and other multi-modal data (text & audio & video)
        self.valid_idx  = list()
        self.test_valid_idx = list()    
        self.file_lst = list()
        
        self.test_ref_clips = []

        std_file = osp.join(self.path ,'std.npy')
        avg_file = osp.join(self.path ,'avg.npy')
        mocap_file = osp.join(self.path ,'mocap.npy')
        valid_idx_file = osp.join(self.path ,'idx.npy')
        test_idx_file = osp.join(self.path, 'test_idx.npy')
        flst_file = osp.join(self.path ,'file_lst.txt')
        label_file = osp.join(self.path ,'label.npy')
        extra_file = osp.join(self.path ,'extra.npz')

        check_label_file = True if (osp.exists(label_file) and self.use_cond) or not self.use_cond else False
        #check_extra_file = True if (osp.exists(extra_file) and self.use_extra_cond) or not self.use_extra_cond else False

        if osp.exists(std_file) and osp.exists(avg_file) and \
            osp.exists(mocap_file) and osp.exists(valid_idx_file) and check_label_file:
        
            self.std = np.load(std_file)
            self.avg = np.load(avg_file)
            self.motion_flattened = np.load(mocap_file)
            self.frame_dim = self.motion_flattened.shape[-1]
            self.valid_idx = np.load(valid_idx_file)
            self.test_valid_idx = np.load(test_idx_file)
            self.file_lst = file_util.read_file_name(flst_file)

            if osp.exists(label_file):
                self.labels= np.load(label_file)

            #if osp.exists(extra_file):
            #    self.extra= np.load(extra_file)

            self.normalization = {
                "mode": 'zscore',
                "avg": self.avg,
                "std": self.std
            }
          
        else:
            file_paths = self.get_motion_fpaths()
            self.total_len = 0
            
            for fname in tqdm.tqdm(file_paths):
                motion = self.process_data(fname)
                
                length = len(motion)

                if self.min_motion_len and length < self.min_motion_len:
                    continue
                if self.max_motion_len != -1 and length > self.max_motion_len:
                    continue

                if self.use_cond:
                    label = self.process_label(fname)
                    self.labels.append(label)

                self.file_lst.append(fname)
                cur_valid_idx = list(range(self.total_len, self.total_len + length - self.rollout))
                cur_test_valid_idx = cur_valid_idx[:-self.test_num_steps]

                self.valid_idx += cur_valid_idx
                self.test_valid_idx += cur_test_valid_idx 
                self.total_len += length
                self.motion_lst.append(motion)
            
            self.frame_dim = motion.shape[-1] 
            self.valid_idx = np.array(self.valid_idx)
            skip_num = len(self.test_valid_idx)//self.test_num_init_frame
            self.test_valid_idx = np.array(self.test_valid_idx)[::skip_num]

            self.motion_flattened = np.concatenate(self.motion_lst,axis=0)
            self.motion_flattened, self.normalization = self.create_norm(self.motion_flattened, 'zscore')
            
            self.std = self.normalization['std']
            self.avg = self.normalization['avg']
            
            if self.use_cond:
                self.labels = np.concatenate(np.array(self.labels),axis=0)
                np.save(label_file,self.labels)

            #if self.use_extra: 
            #    np.save(extra_file,self.extra)

            file_util.save_file_name(flst_file, self.file_lst)
            np.save(std_file,self.std)
            np.save(avg_file,self.avg)
            np.save(valid_idx_file,self.valid_idx)
            np.save(test_idx_file,self.test_valid_idx)
            np.save(mocap_file,self.motion_flattened)
        

        self.test_ref_clips = np.array([self.motion_flattened[idx:idx+self.test_num_steps] for idx in self.test_valid_idx])
        self.frame_size = self.motion_flattened.shape[-1]
        print('ref start index',self.test_valid_idx)
        print('data shape:{}'.format(self.motion_flattened.shape))

    def read_label_data(self, path):
        if self.use_cond:
            raise NotImplementedError("read_label_data: not implemented!")

    def get_motion_fpaths(self, path):
        raise NotImplementedError("path_acq: not implemented!")
        
    
    def x_to_jnts(self, x):
        '''
        transform your customized data form to joints in global space
        x: [N, ...]
        out: [N, J, 3]
        '''
        raise NotImplementedError("process_data: not implemented!")

    
    def x_to_trajs(self, x):
        '''
        transform your customized data form to joints in global space
        x: [N, ...]
        out: [N, J, 3]
        '''
        #raise NotImplementedError("process_data: not implemented!")
        pass

    def process_data(self, fname):
        '''
        take a path as input, output your customized data form
        fname: str
        out: [N, ...]
        '''
        raise NotImplementedError("process_data: not implemented!")
    
    @staticmethod
    def create_norm(mocap_data, norm_mode):
        max = mocap_data.max(axis=0)[0]
        min = mocap_data.min(axis=0)[0]
        avg = mocap_data.mean(axis=0)
        std = mocap_data.std(axis=0)
        std[std == 0] = 1.0
        
        normalization = {
            "mode": norm_mode,
            "max": max,
            "min": min,
            "avg": avg,
            "std": std,
        }

        if norm_mode == "minmax":
            mocap_data = 2 * (mocap_data - min) / (max - min) - 1

        elif norm_mode == "zscore":
            mocap_data = (mocap_data - avg) / std

        else:
            raise ValueError("Unknown normalization mode")
        
        return mocap_data, normalization

    def denorm_data(self,t):
        normalization = self.normalization
        if normalization['mode'] == 'minmax':
            data_max = normalization['max']
            data_min = normalization['min']
            t = (t + 1) * (data_max - data_min) / 2 + data_min
        
        elif normalization['mode'] == 'zscore':
            data_avg = normalization['avg']
            data_std = normalization['std']
            t = t * data_std + data_avg

        else:
            raise ValueError("Unknown normalization mode")
        return t

    def norm_data(self,t):
        normalization = self.normalization
        if normalization['mode'] == 'minmax':
            data_max = normalization['max']
            data_min = normalization['min']
            t = 2 * (t - data_min) / (data_max - data_min) - 1
        
        elif normalization['mode'] == 'zscore':
            data_avg = normalization['avg']
            data_std = normalization['std']
            t = (t - data_avg) / data_std

        else:
            raise ValueError("Unknown normalization mode")
        return t
    
    def get_dim_by_key(self, category, key):
        if category == "heading":
            return [2]
        elif category == "rootdxdy":
            return [0,1]
        
        elif category == "joint_pos":
            return 

        elif category == "joint_angle":
            return 

        elif category == "joint_vel":
            return

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        idx_ = self.valid_idx[idx]
        motion = self.motion_flattened[idx_:idx_+self.rollout]
        return  motion 
