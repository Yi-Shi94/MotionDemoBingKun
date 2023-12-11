from fairmotion.data import bvh
import numpy as np
import copy
import dataset.util.geo as geo_util

def unit_conver_scale(unit):
    if unit in ['feet', 'foot']:
        scale = 1.0/30.48
    elif unit in ['m', 'meter']:
        scale = 1.0/100
    elif unit in ['cm', 'centermeter']:
        scale = 1.0
    else:
        scale = 1.0
        print('unit not implemented, scale as 1.0')
    return scale

def extract_sk_lengths(positions, linked_joints):
    #position: NxJx3
    #single frame rigid body restriction
    lengths = np.zeros((len(linked_joints),positions.shape[0]))
    for i,(st,ed) in enumerate(linked_joints):
        length =  np.linalg.norm(positions[:,st] - positions[:,ed], axis=-1)     
        lengths[i] = length
    return np.mean(lengths,axis=-1)


def get_bvh_frames(path):
    motion = bvh.load(path, load_motion=True)
    positions = motion.positions(local=False) 
    return positions.shape[0]-1


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

def load_bvh_info(bvh_file_path):
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
                channel_num = lineList[1]
                joint_chn_num.append(int(channel_num))
                if joint_parent[-1] == -1:
                    rot_lst = lineList[5:]
                else:
                    rot_lst = lineList[2:]

                joint_rot_order.append(''.join([ob[0] for ob in rot_lst]))

            elif (lineList[0] == "Frame" and lineList[1] == "Time:"):
                frame_time = float(lineList[2])

    joint_offset = np.array(joint_offset).reshape(-1, 3)
    return joint_name, joint_parent, joint_offset, joint_rot_order, joint_chn_num, frame_time


def read_bvh(path, foot_idx_lst, root_idx, unit, source_fps=30, target_fps=30):
    motion = bvh.load(path, load_motion=True)
    positions = motion.positions(local=False)  # (frames, joints, 3)
    orientations = motion.rotations(local=True)
    
    if source_fps > target_fps:
        sample_ratio = int(source_fps/target_fps)
        positions = positions[::sample_ratio]
        orientations = orientations[::sample_ratio]
    
    nfrm, njoint, _ = positions.shape
    
    scale = unit_conver_scale(unit)
    ori = copy.deepcopy(positions[0,root_idx])
    
    y_min = np.min(positions[0,foot_idx_lst,1])
    ori[1] = y_min

    positions = (positions - ori) * scale

    velocities_root = positions[1:,root_idx,:] - positions[:-1,root_idx,:]
    
    positions[:,:,0] -= positions[:,0,:1]
    positions[:,:,2] -= positions[:,0,2:]

    global_heading = - np.arctan2(orientations[:,root_idx,0,2], orientations[:, root_idx, 2,2]) 
    global_heading_diff = (global_heading[1:] - global_heading[:-1]) % (2*np.pi)
    global_heading_rot = np.array([geo_util.rot_yaw(x) for x in global_heading])
    global_heading_rot_inv = global_heading_rot.transpose(0,2,1)

    positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None,:, :], njoint, axis=1), positions[...,None])
    
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1] #np.matmul(np.repeat(global_heading_rot[:-1, None,:, :], njoint, axis=1), (positions[1:] - positions[:-1])[...,None])
    print(velocities_no_heading.shape)
    velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[...,[0,2]]
 
    orientations[:,0,...] = np.matmul(global_heading_rot, orientations[:,0,...]) 

    size_frame = 3+njoint*3+njoint*3+njoint*6
    final_x = np.zeros((nfrm, size_frame))

    final_x[1:,2] = global_heading_diff
    final_x[1:,:2] = velocities_root_xy_no_heading 
    final_x[:,3:3+3*njoint] = np.reshape(positions_no_heading, (nfrm,-1))
    final_x[1:,3+3*njoint:3+6*njoint] = np.reshape(velocities_no_heading, (nfrm-1,-1))
    final_x[:,3+6*njoint:3+12*njoint] = np.reshape(orientations[..., :, :2, :], (nfrm,-1))
    return final_x


if __name__ == '__main__':
    pass
