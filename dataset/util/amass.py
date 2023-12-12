import dataset.util.geo as geo_util
import dataset.util.skeleton_info as skel_dict
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch

def load_amass_file(amass_file, root_idx):
    motion_frame = np.load(amass_file)

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

    
    joints_vel = (joints[1:,...] - joints[:-1,...])
    root_dxdy = joints_vel[:, root_idx,:].clone()
    root_dxdy[...,2] *= 0

    joints[:,:,:2] = joints[:,:,:2] - joints[:,[root_idx],:2]
    root_orient_quat_z_inv = geo_util.quat_conjugate(root_orient_quat_z).float()
    
    for i_jnt in range(joints.shape[1]):
        joints[:,i_jnt,:] = geo_util.quat_rotate(root_orient_quat_z_inv, joints[:,i_jnt,:])
    
    joints_vel = joints[1:,...] - joints[:-1,...]    
    root_dxdy = geo_util.quat_rotate(root_orient_quat_z_inv[:-1], root_dxdy)

    xs = np.concatenate([root_dxdy[:,:2].reshape(root_dxdy.shape[0],-1), 
                        root_drot_angle_z.reshape(root_dxdy.shape[0],-1), 
                        joints[1:].reshape(root_dxdy.shape[0],-1), 
                        joints_vel.reshape(root_dxdy.shape[0],-1), 
                        pose_body[1:].reshape(root_dxdy.shape[0],-1)],axis=-1)  
    return xs

def load_amass_info(amass_file):
    import matplotlib.pyplot as plt
    motion_frame = np.load(amass_file)
    pos = fk_(motion_frame)
    plot_3d_pt(pos)
    return  

def plot_3d_pt(pts):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[0,:,0], pts[0,:,1], pts[0,:,2], marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    print(sss)


def fk(motion_frame):
    
    joint_positions = torch.tensor(motion_frame['joints'])
    dtype = joint_positions.dtype
    num_jnt = joint_positions.shape[1]
    num_frames = joint_positions.shape[0]
    joint_rotations = torch.zeros(num_frames, num_jnt, 4).float()
    joint_orientations = torch.zeros(num_frames, num_jnt, 4).float()
    joint_orientations_inv = torch.zeros(num_frames, num_jnt, 4)
    #fk (at origin)
    joint_offset = torch.zeros(num_frames, num_jnt, 3).float()
    root_rots_expmap = torch.tensor(motion_frame['root_orient'])
    rots_expmap = torch.tensor(motion_frame['pose_body'])
    betas =  motion_frame['betas']
    joint_parent = get_parent_from_link(skel_dict.skel_dict['AMASS']['links'])
    
    for i in range(num_jnt):
        
        if joint_parent[i] == -1: #root
            local_rotation = geo_util.exp_map_to_quat(root_rots_expmap) 
            joint_orientations[:,i] = local_rotation
        else:
            idx = i-1
            local_rotation = geo_util.exp_map_to_quat(rots_expmap[:, 3*idx: 3*idx+3])                
            joint_orientations[:,i] = geo_util.quat_mul(joint_orientations[:,joint_parent[i]], local_rotation)
            joint_positions[:,i] = joint_positions[:,joint_parent[i]] + geo_util.quat_rotate(joint_orientations[:,joint_parent[i]], joint_offset[i])
            
    joint_positions[..., 2] += frames[..., [3]] #height
        
    return joint_positions

def fk_rev(motion_frame):
    
    joint_positions = torch.tensor(motion_frame['joints'])
    dtype = joint_positions.dtype
    num_jnt = joint_positions.shape[1]
    num_frames = joint_positions.shape[0]
    joint_rotations = torch.zeros(num_frames, num_jnt, 4).float()
    joint_orientations = torch.zeros(num_frames, num_jnt, 4).float()
    joint_orientations_inv = torch.zeros(num_frames, num_jnt, 4)
    #fk (at origin)
    joint_offset = torch.zeros(num_frames, num_jnt, 3).float()
    root_rots_expmap = torch.tensor(motion_frame['root_orient'])
    rots_expmap = torch.tensor(motion_frame['pose_body'])
    betas =  motion_frame['betas']
    joint_parent = get_parent_from_link(skel_dict.skel_dict['AMASS']['links'])
    
    for i in range(num_jnt):
        
        if joint_parent[i] == -1: #root
            local_rotation = geo_util.exp_map_to_quat(root_rots_expmap) 
            joint_orientations[:,i] = local_rotation
        else:
            idx = i-1
            local_rotation = geo_util.exp_map_to_quat(rots_expmap[:, 3*idx: 3*idx+3])                
            joint_orientations[:,i] = geo_util.quat_mul(joint_orientations[:,joint_parent[i]], local_rotation)
            #joint_positions[:,i] = joint_positions[:,joint_parent[i]] + geo_util.quat_rotate(joint_orientations[:,joint_parent[i]], joint_offset[i])
            #joint_orientations_inv[:, i] = geo_util.quat_conjugate(joint_orientations[:,i])
            inv_ori = geo_util.quat_conjugate(joint_orientations[:,joint_parent[i]])
            trans = joint_positions[:,i]-joint_positions[:,joint_parent[i]]
            joint_offset[:,i] = geo_util.quat_rotate(inv_ori, trans)
    #joint_positions[..., 1] += frames[..., [4]] #height
    
    return joint_offset

def extract_sk_lengths(positions, linked_joints):
    #position: NxJx3
    #single frame rigid body restriction
    lengths = np.zeros((len(linked_joints),positions.shape[0]))
    for i,(st,ed) in enumerate(linked_joints):
        length =  np.linalg.norm(positions[:,st] - positions[:,ed], axis=-1)     
        lengths[i] = length
    return np.mean(lengths,axis=-1)


def get_parent_from_link(links):
    max_index = -1
    parents_dict = dict()
    parents = list()
    for pair in links:
        st, ed = pair
        if st>ed:
            st, ed = ed, st
            #print(st,ed,max_index)
        max_index = ed if ed>max_index else max_index
        parents_dict[ed] = st
    parents_dict[0] = -1
    for i in range(max_index+1):
        parents.append(parents_dict[i])
    return parents