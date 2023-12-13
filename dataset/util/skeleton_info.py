# SMPL
SMPL_links = [
            (0, 1), 
            (1, 4), 
            (4, 7), 
            (7, 10), 
            (0, 2), 
            (2, 5), 
            (5, 8), 
            (8, 11), 
            (0, 3), 
            (3, 6), 
            (6, 9), 
            (9, 14), 
            (14, 17),
            (17, 19), 
            (19, 21), 
            (9, 13), 
            (13, 16), 
            (16, 18), 
            (18, 20), 
            (9, 12), 
            (12, 15)
            ]

SMPL_name_joint = [
     "root",
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
]



SMPL_joint_offset = [
    [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
    [ 7.4191e-02, -8.5461e-02, -1.7308e-02],
    [-7.1594e-02, -8.3623e-02, -1.4339e-02],
    [-1.3948e-03,  9.8195e-02, -1.4825e-02],
    [ 3.5133e-02, -3.9267e-01,  2.7427e-03],
    [-4.0115e-02, -3.9730e-01,  2.0664e-04],
    [ 1.9905e-03,  1.2180e-01,  1.6719e-04],
    [-1.3309e-02, -4.1153e-01, -4.8053e-02],
    [ 1.4754e-02, -4.0995e-01, -4.7991e-02],
    [ 2.4568e-03,  5.0603e-02,  2.4683e-02],
    [ 2.0389e-02, -5.7741e-02,  1.2779e-01],
    [-2.6866e-02, -4.7960e-02,  1.3085e-01],
    [ 1.4684e-03,  2.3297e-01, -6.0496e-02],
    [ 7.2725e-02,  1.2933e-01, -4.3755e-02],
    [-7.3775e-02,  1.2371e-01, -4.9582e-02],
    [ 8.9063e-03,  6.4604e-02,  5.7413e-02],
    [ 8.6998e-02,  3.8014e-02, -2.4794e-03],
    [-9.2000e-02,  3.9984e-02, -6.2488e-03],
    [ 2.5586e-01, -1.8661e-02, -2.6340e-02],
    [-2.5963e-01, -1.9262e-02, -1.8788e-02],
    [ 2.5177e-01,  1.0090e-02,  2.5703e-03],
    [-2.5894e-01,  4.9262e-03, -2.5922e-03]
]



# MVAE from EA
MVAE_links = [
        [12, 0],  # right foot
        [16, 12],  # right shin
        [14, 16],  # right leg
        [15, 17],  # left foot
        [17, 13],  # left shin
        [13, 1],  # left leg
        [5, 7],  # right shoulder
        [7, 10],  # right upper arm
        [10, 20],  # right lower arm
        [6, 8],  # left shoulder
        [8, 9],  # left upper arm
        [9, 21],  # left lower arm
        [3, 18],  # torso
        [14, 15],  # hip
        ]

MVAE_name_joint = [
    #"root", "hip","rhip","lowerback","lknee","rknee",
    #"upperback","lankle","rankle","chest",
    #"ltoe","rtoe","lowerneck","lclavicle","rclavicle",
    #"upperneck","lshoulder","rshoulder",
    #"lelbow","relbow","lwrist","rwrist"
]

# LAFAN1
LAFAN1_links = [[0,1],
                [1,2],
                [2,3],
                [3,4],
                [0,5],
                [5,6],
                [6,7],
                [7,8],
                [0,9],
                [9,10],
                [10,11],
                [11,12],
                [12,13],
                [11,14],
                [14,15],
                [15,16],
                [16,17],
                [11,18],
                [18,19],
                [19,20],
                [20,21]]


LAFAN1_name_joint= ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe', 
                        'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe', 
                        'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 
                        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
                        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']

LAFAN1_transmap = {-1:-1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 
                6: 5, 7: 6, 8: 7, 9: 8, 10: 8, 
                11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 
                16: 13, 17: 14, 18: 15, 19: 16, 20: 17, 
                21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 21}

LAFAN1_rev_transmap = {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:6, 6:7, 7:8, 8:9, 9:11, 10:12, 11:13,
                    12:14, 13:15, 14:17, 15:18, 16:19, 17:20, 18:22, 19:23, 20:24, 21:25}

LAFAN1_joint_offset = [[ 256.188995, 92.282494, -386.530701],
[ 0.103457,  1.857843,  10.548491],
[ 4.35000000e+01, -3.80000000e-05, -4.00000000e-06],
[ 4.23722000e+01,  8.00000000e-06, -2.10000000e-05],
[ 1.73000030e+01,  1.00000000e-06, -1.10000000e-05],

[ 1.03456000e-01,  1.85781700e+00, -1.05485100e+01],
[ 4.35000380e+01, -1.50000000e-05,  0.00000000e+00],
[ 4.23722610e+01, -1.10000000e-05,  1.30000000e-05],
[ 1.73000030e+01, -2.00000000e-06,  6.00000000e-06],

[ 6.90196700e+00, -2.60372900e+00, -5.00000000e-06],
[ 1.25880990e+01,  6.00000000e-06, -9.00000000e-06],
[ 1.23432040e+01, -5.00000000e-06,  1.70000000e-05],
[ 2.58328900e+01, -8.00000000e-06, -1.00000000e-05],
[ 1.17666130e+01,  1.10000000e-05,  7.00000000e-06],

[ 1.97459070e+01, -1.48037000e+00,  6.00009900e+00],
[ 1.12841290e+01, -1.00000000e-06, -2.20000000e-05],
[ 3.30000340e+01, -1.30000000e-05,  2.00000000e-05],
[ 2.52000100e+01,  2.90000000e-05,  1.20000000e-05],

[ 1.97460980e+01, -1.48036600e+00, -6.00007300e+00],
[ 1.12841420e+01, -2.10000000e-05, -1.00000000e-05],
[ 3.30000990e+01,  2.50000000e-05,  1.00000000e-05],
[ 2.51997720e+01,  1.39000000e-04,  4.18000000e-04]]

#100STYLE
STYLE100_links = [[0, 1],  # hips-chest
                    [1, 2],  # chest-chest2
                    [2, 3],  # chest2-chest3
                    [3, 4],  # chest3-chest4
                    [4, 5],  # chest4-neck
                    [5, 6],  # neck-head 
                    [4, 7],  # neck-rightCollar 
                    [7, 8],  # rightCollar-rightShoulder
                    [8, 9],   # rightShoulder - rightElbow
                    [9, 10],  # rightElbow - rightWrist
                    [4, 11],  # neck-leftCollar
                    [11, 12], # leftCollar-leftShoulder
                    [12, 13],  #leftShoulder-leftElbow
                    [13, 14],  #leftElbow - leftWrist
                    [0, 15],   # hips - rightHip
                    [15, 16],  # rightHip - rightKnee
                    [16, 17],  # rightKnee - rightAnkle
                    [17, 18],  # rightAnkle - rightToe
                    [0, 19],   # hips - leftHip
                    [19, 20],  # leftHip - leftKnee
                    [20, 21],  # leftKnee - leftAnkle
                    [21, 22]]  # leftAnkle - leftToe


STYLE100_name_joint = ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head', 
                        'RightCollor', 'RightShoulder', 'RightElbow', 'RightWrist', 
                        'LeftCollor', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
                        'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
                        'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe']

STYLE100_joint_offset = [[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
 [ 0.0000000e+00,  1.2953391e+01, -2.8419000e-02],
 [ 0.0000000e+00,  1.0280254e+01,  5.1622000e-02],
 [ 0.0000000e+00,  9.2871800e+00,  0.0000000e+00],
 [ 0.0000000e+00,  9.2871800e+00,  0.0000000e+00],
 [ 0.0000000e+00,  1.3296264e+01,  0.0000000e+00],
 [ 0.0000000e+00,  8.9592640e+00,  9.9003000e-02],

 [-3.1935110e+00,  7.4513880e+00,  0.0000000e+00],
 [-1.5792092e+01,  0.0000000e+00,  0.0000000e+00],
 [-3.1151839e+01,  0.0000000e+00,  0.0000000e+00],
 [-2.5536298e+01,  0.0000000e+00,  2.3210000e-03],
 
 [ 3.1935110e+00,  7.4513880e+00,  0.0000000e+00],
 [ 1.5792092e+01,  0.0000000e+00,  0.0000000e+00],
 [ 3.1151839e+01,  0.0000000e+00,  0.0000000e+00],
 [ 2.5536298e+01,  0.0000000e+00,  2.3210000e-03],

 [-1.0745749e+01,  1.3925200e-01,  1.4209000e-02],
 [ 0.0000000e+00, -4.3743899e+01, -6.0710000e-03],
 [ 0.0000000e+00, -4.3214199e+01,  8.9931000e-02],
 [ 0.0000000e+00, -9.6998300e+00,  1.9151588e+01],

 [ 1.0745749e+01,  1.3925200e-01,  1.4209000e-02],
 [ 0.0000000e+00, -4.3743899e+01, -6.0710000e-03],
 [ 0.0000000e+00, -4.3214199e+01,  8.9931000e-02],
 [ 0.0000000e+00, -9.6998300e+00,  1.9151588e+01]]


skel_dict = {
            'STYLE100':{'links':STYLE100_links, 'name_joint':STYLE100_name_joint, 'offset_joint':STYLE100_joint_offset, 'num_joint':23, 'root_idx':0, 'head_idx':[5,6], 'hand_idx':[10,14],'foot_idx':[17,18,21,22],'fps':30,'unit':'meter', 'st_angle_offset':-90},
            'LAFAN1':{'links':LAFAN1_links, 'name_joint':LAFAN1_name_joint, 'offset_joint':LAFAN1_joint_offset, 'num_joint':22, 'root_idx':0, 'head_idx':[12,13],'hand_idx':[17,21],'foot_idx':[3,4,7,8],  'fps':30, 'unit':'meter','st_angle_offset':-180, 'transmap':LAFAN1_transmap, 'rev_transmap':LAFAN1_rev_transmap, },
            'AMASS':{'links': SMPL_links, 'name_joint':SMPL_name_joint,'offset_joint':SMPL_joint_offset, 'num_joint':22, 'root_idx':0,  'head_idx':[12,15], 'hand_idx':[20,21], 'foot_idx':[7,8,10,11], 'fps':30, 'unit':'cm'}
            }

