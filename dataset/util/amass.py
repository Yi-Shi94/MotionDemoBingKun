import dataset.util.geo as geo_util

def load_amass_file(amass_file, root_idx):
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

    origin = copy.deepcopy(joints[0, root_idx,:]) 

    joints_vel = (joints[1:,...] - joints[:-1,...])
    root_dxdy = joints_vel[:, root_idx,:].clone()
    root_dxdy[...,2] *= 0

    joints[:,:,:2] = joints[:,:,:2] - joints[:,[root_idx],:2]
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