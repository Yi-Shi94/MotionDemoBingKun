import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tqdm
import copy
import os
from scipy.spatial.transform import Rotation as R

from .geo_utils import euler_to_matrix
transfer_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 
                6: 5, 7: 6, 8: 7, 9: 8, 10: 8, 
                11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 
                16: 13, 17: 14, 18: 15, 19: 16, 20: 17, 
                21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 21}
rev_transfer_map = {}

for k,v in transfer_map.items():
    v = transfer_map[k]
    if v not in rev_transfer_map:
        rev_transfer_map[v] = k


def get_link(parent):
    link_lst = []
    for idx, idx_par in enumerate(parent):
        if idx_par == -1:
            continue
        link_lst.append([idx,idx_par])
    return link_lst


def load_motion_data(bvh_file_path, num_frames=-1):
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break

        motion_data = []
        frame_idx = 0
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            
            if len(data) == 0:
                break

            if num_frames!=-1 and frame_idx> num_frames:
                break
            
            frame_idx += 1

            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def ik(rot_last, jnt_cur, bvh_info, tol_change, max_iter, device = 'cuda:0'):
    
    joint_name, joint_parent, joint_offset, joint_rot_order = bvh_info
    jnt_cur = torch.tensor(jnt_cur, requires_grad=False, device = device).float()
    joint_offset = torch.tensor(joint_offset, requires_grad=False, device =device).float()
    
    rot_last = torch.tensor(rot_last,device=device,requires_grad=False).float()
    drot = torch.zeros(rot_last.shape[-1], requires_grad=True, device= device)    
    optimizer = optim.LBFGS([drot], 
                    max_iter=max_iter, 
                    tolerance_change=tol_change,
                    line_search_fn="strong_wolfe")

    def f_loss(drot):
        motion_frame = torch.ones(3 + rot_last.shape[-1], device =device).float()
        motion_frame[:3] *= 0
        motion_frame[3:] *= rot_last+drot

        _, joint_positions,  _,   _ = fk_pt(joint_name, joint_parent, joint_offset, joint_rot_order, motion_frame)
        loss = F.mse_loss(joint_positions,jnt_cur) 
        print(loss)
        return loss
    
    def closure():
        optimizer.zero_grad()
        objective = f_loss(drot)
        objective.backward()
        return objective

    optimizer.step(closure)
    rot_cur = rot_last + drot
    return rot_cur.cpu().detach().numpy(), drot.cpu().detach().numpy()


def ik_progressive(rot_init, jnt_seq, bvh_info, tol_change, max_iter, device = 'cuda:0'):
    
    rot_init[:3] = np.array([102.6,8.16,0.879])
    rot_seq = np.zeros((jnt_seq.shape[0],rot_init.shape[-1]))
    rot_last = rot_init

    for i in tqdm.tqdm(range(jnt_seq.shape[0])):
        rot_cur, _ = ik(rot_last, jnt_seq[i], bvh_info, tol_change, max_iter, device)
        rot_last = copy.deepcopy(rot_cur)
        rot_seq[i] = rot_last
        
    return rot_seq


def fk_pt(joint_name, joint_parent, joint_offset, joint_rot_order,  motion_data_cur_frame, device='cuda:0'):
    
    channals_num = (len(motion_data_cur_frame) -3) // 3
    root_position = motion_data_cur_frame[:3]
    rotations = torch.zeros((channals_num, 3)).to(device).float()
    joint_rotations = torch.zeros((channals_num, 3, 3)).to(device).float()
    for i in range(channals_num):
        rot_order = joint_rot_order[i]
        rotations[i] = motion_data_cur_frame[3+3*i: 6+3*i]
        joint_rotations[i] = euler_to_matrix(rot_order, rotations[i,:3], degrees=True)#.as_matrix()
    
    cnt = 0
    num_jnt = len(joint_name)
    joint_positions = torch.zeros((num_jnt, 3)).to(device).float()
    joint_orientations = torch.zeros((num_jnt, 3, 3)).to(device).float()
    joint_offset = joint_offset.float()
    
    for i in range(num_jnt):
        rot_order = joint_rot_order[i]
        if joint_parent[i] == -1: #root
            joint_positions[i] = root_position
            joint_orientations[i] = euler_to_matrix(rot_order, rotations[i,:3], degrees=True)#.as_matrix()
            #joint_orientations[i] = np.matmul(offset_mat,cur_joint_orientations)
        else:
            if "_end" not in joint_name[i]:     # 末端没有CHANNELS
                cnt += 1
            r = euler_to_matrix(rot_order, rotations[cnt,:3], degrees=True)
            joint_orientations[i] = torch.matmul(joint_orientations[joint_parent[i]].clone(), r)
            joint_positions[i] = joint_positions[joint_parent[i]] + torch.matmul(joint_orientations[joint_parent[i]].clone(),joint_offset[i])
    return  root_position, joint_positions, joint_orientations, joint_rotations


def fk(joint_name, joint_parent, joint_offset, joint_rot_order, motion_data_cur_frame, offset_mat=np.eye(3)):
    
    channals_num = (len(motion_data_cur_frame) -3) // 3
    root_position = motion_data_cur_frame[:3]
    rotations = np.zeros((channals_num, 3), dtype=np.float64)
    joint_rotations = np.zeros((channals_num, 3, 3), dtype=np.float64)
    
    for i in range(channals_num):
        rot_order = joint_rot_order[i]
        rotations[i] = motion_data_cur_frame[3+3*i: 6+3*i]
        mat =  R.from_euler(rot_order, [rotations[i][0], rotations[i][1], rotations[i][2]], degrees=True).as_matrix()
        joint_rotations[i] = mat
    
    cnt = 0
    num_jnt = len(joint_name)
    joint_positions = np.zeros((num_jnt, 3), dtype=np.float64)
    joint_orientations = np.zeros((num_jnt, 3, 3), dtype=np.float64)
    
    for i in range(num_jnt):
        
        rot_order = joint_rot_order[i]
        if joint_parent[i] == -1: #root
            joint_positions[i] = root_position
            cur_joint_orientations = R.from_euler(rot_order, [rotations[cnt][0], rotations[cnt][1], rotations[cnt][2]], degrees=True).as_matrix()
            joint_orientations[i] = np.matmul(offset_mat,cur_joint_orientations)

        else:
            if "_end" not in joint_name[i]:     # 末端没有CHANNELS
                cnt += 1
            r = R.from_euler(rot_order, [rotations[cnt][0], rotations[cnt][1], rotations[cnt][2]], degrees=True)
            joint_orientations[i] = (R.from_matrix(joint_orientations[joint_parent[i]]) * r).as_matrix()
            joint_positions[i] = joint_positions[joint_parent[i]] + R.from_matrix(joint_orientations[joint_parent[i]]).apply(joint_offset[i])
    return  root_position, joint_positions, joint_orientations, joint_rotations



def retarget_pose(anchor_orientation, poses, joint_parent, joint_rot_order):
    #channel_num = poses.shape[-1] // 3 -1  #njnt+1
    new_poses = np.zeros_like(poses)
    new_poses[:,:3] = poses[:,:3]
    num_seq = poses.shape[0]

    # Stationary: Q _ L pose-> T pose 
    Q_l2t_lst = anchor_orientation 
    Q_l2t_inv_lst = np.zeros((len(joint_parent),3,3))
    for j in range(len(joint_parent)):
        Q_l2t_inv_lst[j] = anchor_orientation[j].transpose(1,0)
    
    for n in range(num_seq):
        i_chn = -1.5
        for j in range(len(joint_parent)):
            if transfer_map[j] == i_chn:
                continue
            i_chn = transfer_map[j]
            eur_l = poses[n,(3+i_chn*3):(3+(i_chn+1)*3)]
            #if j == 6:
            #    from IPython import embed; embed()
            R_cur_l = R.from_euler(joint_rot_order[0], [eur_l[0], eur_l[1], eur_l[2]], degrees=True).as_matrix()
            Q_cur_l2t_inv = Q_l2t_inv_lst[j]

            if joint_parent[j] == -1:
                R_cur_t = np.dot(R_cur_l, Q_cur_l2t_inv)

            else:    
                Q_par_l2t = Q_l2t_lst[joint_parent[j]]
                R_cur_t = np.dot(np.dot(Q_par_l2t, R_cur_l), Q_cur_l2t_inv)
                
            new_euler = R.from_matrix(R_cur_t).as_euler(joint_rot_order[0], degrees=True)        
            new_poses[n,(3+i_chn*3):(3+(i_chn+1)*3)] = new_euler
    return new_poses

    
def reform_norm_skel(anchor_pose, old_offset, joint_name, joint_parent, joint_rot_order):
    _, joint_positions, joint_orientations, joint_rotations = \
            fk(joint_name, joint_parent, old_offset, joint_rot_order, anchor_pose)
    
    new_offset = np.zeros_like(old_offset)
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            offset = old_offset[i]
            continue
        parent_idx = joint_parent[i]
        offset = joint_positions[i] - joint_positions[parent_idx]
        new_offset[i] = offset

    root_idx = joint_parent.index(-1)
    new_offset[root_idx,:] = joint_positions[root_idx,:]
    
    return new_offset, joint_orientations


def load_joint_info(bvh_file_path):
    joint_name = []
    joint_parent = []
    joint_offset = []
    joint_rot_order = []
    joint_chn_num = []

    cnt = 0
    myStack = []
    root_joint_name = None
    frame_time = 0 
    
    with open(bvh_file_path, 'r') as file_obj:
        for line in file_obj:
            lineList = line.split()
            if (lineList[0] == "{"):
                myStack.append(cnt)
                cnt += 1

            if (lineList[0] == "}"):
                myStack.pop()

            if (lineList[0] == "OFFSET"):
                joint_offset.append([float(lineList[1]), float(lineList[2]), float(lineList[3])])

            if (lineList[0] == "JOINT"):
                joint_name.append(lineList[1])
                joint_parent.append(myStack[-1])
            
            elif (lineList[0] == "ROOT"):
                joint_name.append(lineList[1])
                joint_parent.append(-1)
                root_joint_name = lineList[1]

            elif (lineList[0] == "End"):
                joint_name.append(joint_name[-1] + '_end')
                joint_parent.append(myStack[-1])
                joint_rot_order.append(joint_rot_order[-1])

            elif (lineList[0] == "CHANNELS"):
                #joint_name.append(joint_name[-1] + '_end')
                #joint_parent.append(myStack[-1])
                channel_num = lineList[1]
                joint_chn_num.append(int(channel_num))
                if joint_parent[-1] == -1:
                    rot_lst = lineList[5:]
                else:
                    rot_lst = lineList[2:]

                joint_rot_order.append(''.join([ob[0] for ob in rot_lst]))

            elif (lineList[0] == "Frame" and lineList[1] == "Time:"):
                fps = int(1/float(lineList[2]))

    joint_offset = np.array(joint_offset).reshape(-1, 3)
    #joint_offset[joint_name.index(root_joint_name)] *= 0
    return joint_name, joint_parent, joint_offset, joint_rot_order, joint_chn_num, frame_time


def output_bvh(out_bvh_path, ori_bvh_path, motion, joint_name, offset):
    print('outputing data:')
    frm = len(motion)    
    bone_info_endl = -1
    ori_kept_line = []
    
    with open(ori_bvh_path, 'r') as file_obj:
        cur_jnt_name = None
        num_tab = 0
        for i, line in enumerate(file_obj):
            
            lineList = line.strip().split()

            if lineList[0] == 'Frame' and lineList[1] == 'Time:':
                ori_kept_line.append(line)
                bone_info_endl = i
                print(i, lineList, 'ended')
                break
            
            elif lineList[0] == 'Frames:':
                ori_kept_line.append('Frames: {}\n'.format(frm))

            elif lineList[0] == '{':
                ori_kept_line.append('\t'*num_tab+'{\n')
                num_tab += 1

            elif lineList[0] == '}':
                num_tab -= 1
                ori_kept_line.append('\t'*num_tab+'}\n')

            elif lineList[0] in ['ROOT','JOINT']:
                cur_jnt_name = lineList[1]
                ori_kept_line.append(line)

            elif (len(lineList)>1 and lineList[0]== 'End' and lineList[1]== 'Site'):
                cur_jnt_name = cur_jnt_name + '_end'
                ori_kept_line.append(line)


            elif lineList[0] == 'OFFSET':
                if offset is None:
                    ori_kept_line.append(line)
                    continue

                cur_offset = offset[joint_name.index(cur_jnt_name)]
                ori_kept_line.append('\t'*num_tab+'OFFSET {:.6f} {:.6f} {:.6f}\n'.format(cur_offset[0],cur_offset[1],cur_offset[2]))

            elif lineList[0] == 'MOTION':
                bone_info_endl = i
                ori_kept_line.append(line)

            else:
                ori_kept_line.append(line)

    with open(out_bvh_path, 'w') as file_obj:
        for line in ori_kept_line:
            file_obj.write(line)

        for i, rot_vec in enumerate(motion):
            rot_str = ' '.join([str('{:.6f}'.format(x)) for x in list(rot_vec)])+'\n'
            file_obj.write(rot_str)
    return

def dxdydth_to_traj(dxdydr, scale=1):
    def rot(yaw):
        cs = np.cos(yaw)
        sn = np.sin(yaw)
        return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])
    
    dxdy = dxdydr[:,:2]
    dr = dxdydr[:,2] 
    dpm = np.array([[0.0,0.0,0.0]])
    dpm_lst = np.zeros((dxdy.shape[0],3))
    yaws = np.cumsum(dr)
    yaws = yaws - (yaws//(np.pi*2))*(np.pi*2)
    rots = np.zeros((dxdy.shape[0],3,3))

    for i in range(1, dxdy.shape[0]):
        rot_yaw = rot(yaws[i])
        rots[i] = rot_yaw
        cur_pos = np.zeros((1,3))
        cur_pos[0,0] = dxdy[i,0]
        cur_pos[0,2] = dxdy[i,1]
        dpm += np.dot(cur_pos,rot_yaw)
        dpm_lst[i,:] = copy.deepcopy(dpm)

    return dpm_lst, rots

def ik_x_to_outmot(x, root_idx, jnt_num, bvh_ori_file, bvh_out_file, st_idx=None, st_frame=None, ik=True,x_flip=False):
    def jnt_conv(new_jnt_num, old_jnt, transfer_map):
        # for mapping jnt positions in different order
        N = old_jnt.shape[0]
        new_jnt = np.zeros((N, new_jnt_num, 3))
        for i in range(new_jnt_num):
            new_jnt[:,i] = old_jnt[:,transfer_map[i]]
        return new_jnt
    
    joint_name, joint_parent, joint_offset, joint_rot_order, _ , _ = load_joint_info(bvh_ori_file)
    bvh_info = [joint_name, joint_parent, joint_offset, joint_rot_order]
    dxdydth = x[:,:3]
    joint_pos = x[:,3:3+jnt_num*3]
    
    trajs_3d, rots = dxdydth_to_traj(dxdydth)

    assert st_idx is not None or st_frame is not None
    if st_frame is None:
        rot_init = load_motion_data(bvh_ori_file)[st_idx,3:]
    else:
        rot_init = st_frame[3:]

    joint_pos = joint_pos.reshape(-1, jnt_num, 3)
    
    heights = joint_pos[:,0,1] * 30.48

    for i in range(rots.shape[0]):
        joint_pos[i] = np.dot(joint_pos[i], rots[i])

    joint_pos *= 30.48
    trajs_3d *= 30.48

    if x_flip:
        joint_pos[:,:,0] = -joint_pos[:,:,0]
        trajs_3d[:,0] = -trajs_3d[:,0]

    joint_pos = joint_pos + trajs_3d[:,None,...]
    joint_pos[:,:,1] -= np.min(joint_pos[:1,:,1])
    joint_pos -= joint_pos[:1,:1,:]
    joint_pos_ = joint_pos - joint_pos[:,:1,:] 


    new_joint_pos = jnt_conv(27, joint_pos_, transfer_map)

    out_motion = np.zeros((x.shape[0]-1, rot_init.shape[-1]+3))
    out_motion_  = ik_progressive(rot_init, new_joint_pos[1:], bvh_info, max_iter=200, tol_change=0.0001)

    out_motion[:,:3] = trajs_3d[1:,:]
    out_motion[:,3:] = out_motion_
    out_motion[:,1] = heights[1:]

    joint_offset = torch.from_numpy(joint_offset).float()
    ik_output = []

    for  i in range(1, out_motion.shape[0]):
        out_mot = torch.from_numpy(out_motion[i]).float()
        _, ik_points,  _,   _ = fk_pt(joint_name, joint_parent, joint_offset, joint_rot_order, out_mot, device='cpu')
        ik_output.append(ik_points.numpy())
    
    ik_output = np.array(ik_output)
    ik_output = jnt_conv(22, ik_output, rev_transfer_map)
    vis_skel(ik_output, LEFAN1_links,save_path=bvh_out_file+'/fked')
    output_bvh(bvh_out_file+'/out.bvh', bvh_ori_file, out_motion, joint_name, joint_offset)

def pose_fix_LAFAN(motion_seqs, fname_in, f_name_out, smooth_level):
    
    joint_name, joint_parent, joint_offset, joint_rot_order, joint_chn_num, frame_time = load_joint_info(fname_in)
    motion = load_motion_data(fname_in)
    anchor_motion = motion[0]
    #print(anchor_motion[3:6])
    #print(anchor_motion[:6])
    #height_off = anchor_motion[1]
    anchor_motion[[0,2]] *= 0
    #anchor_motion[0] = height_off
    #print(anchor_motion[:6])
    #print(anchor_motion[3:6])
    new_offset, anchor_orientation = reform_norm_skel(anchor_motion, joint_offset, joint_name, joint_parent, joint_rot_order)
    new_motion = retarget_pose(anchor_orientation, motion_seqs, joint_parent, joint_rot_order)
    #new_offset[0,:] *= 0
    if smooth_level > 0:
        new_motion = moving_average(new_motion,smooth_level)
    
    output_bvh(f_name_out, fname_in, new_motion, joint_name, new_offset)

def euler_to_quat(euler_angle, order='ZYX'):
    return  R.from_euler(order, euler_angle, degrees=True).as_quat()

def quat_to_euler(quat, order='ZYX'):
    return  R.from_quat(quat).as_euler(order, degrees=True)

def euler_to_m6d(euler_angle, order='ZYX'):
    return  R.from_euler(order, euler_angle, degrees=True).as_matrix()[:2].flatten()


def m6d_to_euler(m6d, order='ZYX'):
    mat = np.zeros((3,3))
    a1 = m6d[:3] / np.linalg.norm(m6d[:3])
    a2 = m6d[3:6]
    a2 = a2-(a1*a2).sum()*a1
    a2 = a2 / np.linalg.norm(a2)
    a3 = np.cross(a1,a2)
    mat[0] = a1
    mat[1] = a2
    mat[2] = a3
    return  R.from_matrix(mat).as_euler(order, degrees=True)

def moving_average(motion, window_size, mode='6d'):
    num_seq, num_dim = motion.shape
    num_chn = num_dim // 3 -1
    
    if mode=='6d':
        mid_dim = 6
        euler_to_mid = euler_to_m6d
        mid_to_euler = m6d_to_euler

    elif mode=='quat':
        mid_dim = 4
        euler_to_mid = euler_to_quat
        mid_to_euler = m6d_to_euler

    motion_qrt = np.zeros((motion.shape[0],mid_dim*22+3))
    motion_qrt_out = np.zeros((motion.shape[0],mid_dim*22+3))

    motion_qrt[:,:3] = motion[:,:3]
    motion_out = np.zeros((motion.shape[0], num_dim))
    motion_out[:,:3] = motion[:,:3]
    
    for i in range(num_seq):
        for j in range(num_chn):
            input = motion[i,(3+3*j):(3+3*(j+1))]
            motion_qrt[i,(3+mid_dim*j):(3+mid_dim*(j+1))] = euler_to_mid(input)

    for j in range(motion_qrt.shape[-1]):
        window = np.ones(window_size) / window_size
        motion_qrt_out[:,j] = np.convolve(motion_qrt[:,j], window, mode='same')
        
    for i in range(num_seq):
        for j in range(num_chn):
            quat = motion_qrt_out[i,(3+mid_dim*j):(3+mid_dim*(j+1))]
            motion_out[i,(3+3*j):(3+3*(j+1))] = mid_to_euler(quat)
    return motion_out


def keyframe_from_bvh(fn, fout_dir, frame_skip=20):
    joint_name, joint_parent, joint_offset, _,_,_ = load_joint_info(fn) 
    motion = load_motion_data(fn)
    
    frame_idxs = [i for i in range(0,motion.shape[0],frame_skip)]
    print(len(frame_idxs),'/',motion.shape[0])
    for i in range(len(frame_idxs)):
        idx = frame_idxs[i]
        motion_frame = [motion[idx]]
        f_name_out = os.path.join(fout_dir,str(i)+'.bvh')
        output_bvh(f_name_out, fn, motion_frame, joint_name, joint_offset)

def get_traj_bvh(file):
    outmotion = load_motion_data(file)
    xyz = outmotion[:,:3]
    xy = xyz[:,[0,2]]
    return xy

def get_traj_x(x):
    def rot(yaw):
        cs = np.cos(yaw)
        sn = np.sin(yaw)
        return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])

    cur_pos = np.array([[0.0,0.0,0.0]])
    cum_x  = np.array([[0.0,0.0,0.0]])

    traj = np.zeros((x.shape[0],3))
    dr = x[:,2]
    yaws = np.cumsum(dr)
    yaws = yaws - (yaws//(np.pi*2))*(np.pi*2)


    for i in range(1,x.shape[0]):
        
        cur_pos = np.zeros((1,3))
        cur_pos[0,0] = x[i,0]
        cur_pos[0,2] = x[i,1]
        cum_x += np.dot(cur_pos,rot(yaws[i]))
        traj[i,:] = copy.deepcopy(cum_x)
    return traj[:,[0,2]]


if __name__=='__main__':
    seqs = load_motion_data('data/train/aiming2_subject5.bvh')[:200]
    basic_fn = 'data/train/run1_subject5.bvh'
    pose_fix_LAFAN(seqs, basic_fn, 'data/aiming2_subject5_tpose_or.bvh', smooth_level=1)