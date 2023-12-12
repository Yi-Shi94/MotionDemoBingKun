import copy
import glob
import torch
import torch.optim as optim
import tqdm
import numpy as np

import dataset.base_dataset as base_dataset
import dataset.util.plot as plot_util
import dataset.util.geo as geo_util
import os.path as osp

class AMASS(base_dataset.BaseMotionData):
    NAME = 'AMASS'
    def __init__(self, config):
        super().__init__(config)
        
    def x_to_jnts(self, x):
        dang = x[:,2]
        dxdy = x[:,:2]
        dxdydz = np.zeros((x.shape[0],3))
        dxdydz[:,:2] = dxdy
        
        joints = x[:,3:69].reshape((x.shape[0],22,3))
        out_joints = np.zeros((x.shape[0], 22, 3))
        cur_dxdydz = np.array([[0.0,0.0,0.0]])
        cur_angle = 0
        for i in range(x.shape[0]):
            cur_angle += dang[i]
            rotmat = geo_util.rot_pitch(cur_angle)
            cur_dxdydz += np.dot(dxdydz[i], rotmat)
            out_joints[i] = np.dot(joints[i], rotmat) + cur_dxdydz
        return out_joints
    
    def plot_jnts(self, x, path=None):
        return plot_util.plot_amass(x, self.links, path)

    def plot_traj(self, x, path=None):
        return plot_util.plot_traj_amass(x, path)

    
    
    def fk(self, joint_angles):
        
        return


    def ik(self, joint_pos):
        
        return 


    def get_motion_fpaths(self):
        path =  osp.join(self.path,'**/*.{}'.format('npz'))
        file_lst = glob.glob(path, recursive = True)
        return file_lst
    
    def process_data(self, fname):
        xs = amass_util.load_amass_file(fname)
        return xs   

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        idx_ = self.valid_idx[idx]
        motion = self.motion_flattened[idx_:idx_+self.rollout]
        return  motion


if __name__=='__main__':
    
    dataset = AMASSDataset(args)
    for data in dataset:
        pass
