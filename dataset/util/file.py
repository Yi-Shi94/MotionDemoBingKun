
def save_file_name(fpath, file_lst):
    with open(fpath,'w') as f:
        for i in range(len(file_lst)):
            f.write(file_lst[i]+'\n')
    return

def read_file_name(fpath):
    file_lst = [] 
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_lst.append(line.strip())
    return file_lst


def prepare_data(path, format, out_path, out_name, csv_dict=None):
    assert format in ['STYLE100','LAFAN1']
   
    if isinstance(path, str):
        if path.split('.')[-1] == 'bvh':
            paths = [path]
        else:
            paths = glob.glob(path+'/*.bvh')
    else:
        paths = path
    idx = []
    poses = []
    print(paths)
    for p in tqdm.tqdm(paths):
        cur_pose = read_bvh(p, skel_dict[format]['foot_idx'],skel_dict[format]['root_idx'],unit_conv='feet')
        if csv_dict is not None:
            assert p in csv_dict
            st,ed = csv_dict[p]
            cur_pose = cur_pose[st:ed]
        nfrm = cur_pose.shape[0]
        idx.append(nfrm)
        poses.append(cur_pose)
    idx = np.cumsum(idx)-1    
    poses = np.concatenate(poses,axis=0) 
    
    if out_path is not None:
        fout = open(os.path.join(out_path,'mocap_file_lst.txt'),'w+')
        for file_name in paths:
            fout.write(file_name+'\n')
        fout.close()

        np.savez(os.path.join(out_path,out_name),data=poses, end_indices=idx)
    return poses,idx

def preprare_seg_STYLE100():
    seg_dict = {}
    dir_dict = {'BR':[1,2],'BW':[3,4],'FR':[5,6],'FW':[7,8],'ID':[9,10],
                'SR':[11,12],'SW':[13,14],'TR1':[15,16],'TR2':[17,18],'TR3':[19,20]}

    seg_path = '../data/STYLE100/Frame_Cuts.csv'
    base_path = '../data/STYLE100/STYLE11'
    with open(seg_path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.split(',')
        file_name = parts[0]
        for key,(st,ed) in dir_dict.items():
            if parts[st] =='N/A':
                continue
            full_name = os.path.join(base_path,file_name,file_name+'_'+key+'.bvh')
            seg_dict[full_name] = [int(parts[st]),int(parts[ed])] 

    return seg_dict

def load_100style(in_path, out_path):
    seg_dict = preprare_seg_STYLE100()
    bvhs = glob.glob(in_path+'/*.bvh')
    poses,idx = prepare_data(bvhs,'STYLE100', out_path=out_path, out_name='mocap_data_part.npz', csv_dict=seg_dict)


def load_LAFAN1(in_path, out_path):
    bvhs = glob.glob(in_path+'/*.bvh')
    poses,idx = prepare_data(bvhs,'LAFAN1', out_path=out_path, out_name='mocap.npz')

