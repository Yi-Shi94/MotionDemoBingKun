import copy
import glob
import torch
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
    
    def fk(self, joint_angles):
        pass


    def ik(self, joint_pos):
        pass


    def get_motion_fpaths(self):
        path =  osp.join(self.path,'**/*.{}'.format('npz'))
        file_lst = glob.glob(path, recursive = True)
        return file_lst
    
    def process_data(self, fname):
        motion_frame = np.load(fname)

        fps = motion_frame['fps']
        gender = motion_frame['gender']
        floor_height = motion_frame['floor_height']
        contacts  = motion_frame['contacts']
        trans = motion_frame['trans']
         
        root_orient = motion_frame['root_orient']
        pose_body = motion_frame['pose_body']
        betas = motion_frame['betas']
        joints = motion_frame['joints']
        joints_vel = motion_frame['joints_vel']

        len_seq = root_orient.shape[0]
        root_heights = copy.deepcopy(joints[:,[0],2])
        headings_angle = np.zeros((len_seq))
        headings_rot = np.zeros((len_seq,3,3))
        skel_pos_vel = np.zeros((len_seq,22,3))
        
        root_orient = torch.tensor(root_orient)
        joints = torch.tensor(joints)
        pose_body = torch.tensor(pose_body)#.view(pose_body.shape[0],-1,3)

        root_orient_quat = geo_util.exp_map_to_quat(root_orient)
        root_orient_quat_z = geo_util.calc_heading_quat(root_orient_quat)
        root_orient_quat_xy = geo_util.quat_diff(root_orient_quat_z, root_orient_quat) 

        rot0 = root_orient_quat_z[..., :-1, :]
        rot1 = root_orient_quat_z[..., 1:, :]

        root_drot_quat_z = geo_util.quat_mul(geo_util.quat_conjugate(rot1), rot0)
        root_drot_angle_z = geo_util.calc_heading(root_drot_quat_z)

        origin = copy.deepcopy(joints[0,self.root_idx,:]) 

        joints_vel = (joints[1:,...] - joints[:-1,...])
        root_dxdy = joints_vel[:,self.root_idx,:].clone()
        root_dxdy[...,2] *= 0

        joints[:,:,:2] = joints[:,:,:2] - joints[:,[self.root_idx],:2]
        root_orient_quat_z_inv = geo_util.quat_conjugate(root_orient_quat_z).float()
        
        for i_jnt in range(joints.shape[1]):
            joints[:,i_jnt,:] = geo_util.quat_rotate(root_orient_quat_z_inv, joints[:,i_jnt,:])
            joints_vel[:,i_jnt,:] = geo_util.quat_rotate(root_orient_quat_z_inv[:-1], joints_vel[:,i_jnt,:] )
        
        root_dxdy = geo_util.quat_rotate(root_orient_quat_z_inv[:-1], root_dxdy)
       
        xs = np.concatenate([root_dxdy[:,:2].reshape(root_dxdy.shape[0],-1), 
                            root_drot_angle_z.reshape(root_dxdy.shape[0],-1), 
                            joints[1:].reshape(root_dxdy.shape[0],-1), 
                            joints_vel[:,1:].reshape(root_dxdy.shape[0],-1), 
                            pose_body[1:].reshape(root_dxdy.shape[0],-1)],axis=-1)
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
