import copy
import glob
import torch
import torch.optim as optim
import tqdm
import numpy as np

import dataset.base_dataset as base_dataset
import dataset.util.amass as amass_util
import dataset.util.plot as plot_util
import dataset.util.geo as geo_util
import dataset.util.unit as unit_util
import os.path as osp

class AMASS(base_dataset.BaseMotionData):
    NAME = 'AMASS'
    def __init__(self, config):
        super().__init__(config)
        self.joint_offset = unit_util.unit_conver_scale(self.unit)*np.array(self.skel_info["offset_joint"])
        self.joint_parent = amass_util.get_parent_from_link(self.skel_info["links"])
        self.joint_name = self.skel_info["name_joint"]

    def plot_jnts(self, x, path=None):
        return plot_util.plot_amass(x, self.links, path)

    def plot_traj(self, x, path=None):
        return plot_util.plot_traj_amass(x, path)
    
    def get_motion_fpaths(self):
        path =  osp.join(self.path,'**/*.{}'.format('npz'))
        file_lst = glob.glob(path, recursive = True)
        return file_lst
    
    def process_data(self, fname):
        xs = amass_util.load_amass_file(fname, self.root_idx)
        amass_util.load_amass_info(fname)
        return xs   

    def ik_seq_slow(self, init_frame, frames, max_iter=1000, tol_change=0.0000001, device = 'cuda:0'):
        num_jnt = len(self.joint_name)
        init_rotation = init_frame[..., 3+6*num_jnt: 3+12*num_jnt]
        frames = copy.deepcopy(frames)
        positions = frames[..., 3:3+3*num_jnt]
        positions = positions.reshape((-1, num_jnt, 3))
        positions[...,2] -= frames[...,None,5]

        last_rotation = init_rotation.reshape(-1, num_jnt, 6)
        rotations = np.zeros((frames.shape[0], num_jnt, 6))

        for i in tqdm.tqdm(range(positions.shape[0])):
            cur_rotation = self.ik_frame_slow(last_rotation, positions[i], max_iter, tol_change, device)
            last_rotation = copy.deepcopy(cur_rotation)
            rotations[i] = last_rotation
        return rotations

    def ik_frame_slow(self, last_rotation, current_positions, max_iter, tol_change, device):
        #last_rotation (num_joint * 6) 
        #current_positions (num_joint * 3)
        torch.autograd.set_detect_anomaly(True)
        last_rotation = torch.tensor(last_rotation, requires_grad=False, device=device).float()
        drot = torch.zeros(*last_rotation.shape, requires_grad=True, device= device).float()
        current_positions = torch.tensor(current_positions, requires_grad=False, device=device).float()
        optimizer = optim.LBFGS([drot], 
                        max_iter=max_iter, 
                        tolerance_change=tol_change,
                        line_search_fn="strong_wolfe")

        def f_loss(drot):
            fk_joint_positions = self.fk_local_frame_pt(last_rotation + drot)
            loss = torch.nn.functional.mse_loss(fk_joint_positions, current_positions)
            print(loss)
            return loss
        
        def closure():
            optimizer.zero_grad()
            objective = f_loss(drot)
            objective.backward()
            return objective

        optimizer.step(closure)
        cur_rotation = last_rotation + drot
        drot = drot.cpu().detach().numpy()
        return cur_rotation.cpu().detach().numpy()

    def fk_local_frame_pt(self, rotation_6d):
        num_jnt = len(self.joint_name)
        joint_orientations = torch.zeros((num_jnt, 3, 3), device=rotation_6d.device, dtype=rotation_6d.dtype)
        joint_positions = torch.zeros((num_jnt,3), device=rotation_6d.device, dtype=rotation_6d.dtype)

        joint_offset_pt = torch.tensor(self.joint_offset, device = rotation_6d.device, requires_grad=False, dtype=rotation_6d.dtype)
        for i in range(num_jnt):
            local_rotation = geo_util.m6d_to_rotmat(rotation_6d[..., i,:])
            if self.joint_parent[i] == -1: #root
                joint_orientations[i,:,:] = local_rotation 
            else:                
                joint_orientations[i] = torch.matmul(joint_orientations[self.joint_parent[i]].clone(), local_rotation)
                joint_positions[i] = joint_positions[self.joint_parent[i]] + torch.matmul(joint_orientations[self.joint_parent[i]].clone(), joint_offset_pt[i])
        
        return joint_positions#.view(-1)


    def fk_local_seq(self, frames):
        dtype = frames.dtype
        num_jnt = len(self.joint_name)
        num_frames = len(frames)
        cnt = 0

        rot_frames = frames[:,3+num_jnt*6:]
        joint_offset = np.array(self.joint_offset)

        joint_positions = np.zeros((num_frames, num_jnt, 3))
        joint_orientations = np.zeros((num_frames, num_jnt, 3, 3))
        #fk (at origin)
        for i in range(num_jnt):
            local_rotation = geo_util.rotation_6d_to_matrix(rot_frames[:, i*6:i*6+6]) 
            if i == 0: #root
                joint_orientations[:,i] = local_rotation
            else:
                joint_orientations[:,i] = np.matmul(joint_orientations[:,self.joint_parent[i]], local_rotation)
                joint_positions[:,i] = joint_positions[:,self.joint_parent[i]] + np.matmul(joint_orientations[:,self.joint_parent[i]], joint_offset[i])
        
        joint_positions[..., :, 2] += frames[..., [0], 5] #height
        #self.plot_jnts(joint_positions)
        return  joint_positions


    def vel_step_frame(self, last_frame, frame):
        vel = copy.deepcopy(frame[...,3+3*num_jnt:3+6*num_jnt])
        last_pos = last_frame[...,3:3*num_jnt]
        pos = vel + last_pos
        return pos.view(-1,3)

    def vel_step_seq(self, frames):
        num_jnt = len(self.joint_name)
        num_frames = len(frames)
        frames = copy.deepcopy(frames)
        new_positions = np.zeros((num_frames, 3*num_jnt))
        new_positions[0] = frames[0, 3:3+3*num_jnt]  
        st_positions = new_positions[0]
        for i in range(1, new_positions.shape[0]):
            st_positions += frames[i, 3+3*num_jnt:3+6*num_jnt]  
            new_positions[i, :] = st_positions   
        new_positions = new_positions.reshape((-1, num_jnt,3))
        return new_positions

    def joint_step_frame(self, frame):
        jnts =  copy.deepcopy(frame[...,3:3*self.num_jnt+3])
        jnts = jnts.reshape(-1,self.num_jnt,3)
        return jnts

    def joint_step_seq(self, frames):
        jnts =  copy.deepcopy(frames[...,3:3*self.num_jnt+3])
        jnts = jnts.reshape(-1,self.num_jnt,3)
        return jnts


    def x_to_jnts(self, x, mode):
        x = copy.deepcopy(x)
        dr = x[:,2]
        dxdydz = np.zeros((x.shape[0],3))
        dxdydz[:,:2] = x[:,:2]
        num_jnt = len(self.joint_name)
        if mode == 'angle':
            joints = self.fk_local_seq(x) 
        elif mode == 'position':
            joints = self.joint_step_seq(x)
        elif mode == 'velocity':
            joints = self.vel_step_seq(x)
        elif mode == 'ik_fk':
            rotations = self.ik_seq_slow(x[0],x[1:])
            x[1:, 3+6*num_jnt:3+12*num_jnt] = rotations.reshape(-1,6*num_jnt)
            joints = self.fk_local_seq(x)

         
        out_joints = np.zeros((x.shape[0], 22, 3))
        cur_dxdydz = np.array([[0.0,0.0,0.0]])
        cur_angle = 0

        rs = np.cumsum(dr)
        rs = rs - (rs//(np.pi*2))*(np.pi*2)
        for i in range(x.shape[0]):
            cur_angle = rs[i]
            rotmat = geo_util.rot_pitch(cur_angle)
            cur_dxdydz += np.dot(dxdydz[i], rotmat)
            out_joints[i] = np.dot(joints[i], rotmat) + cur_dxdydz
        return out_joints
    

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
