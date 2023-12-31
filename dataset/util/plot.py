
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
import matplotlib.cm as cm

def plot_traj_amass(clips, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = cm.get_cmap('Spectral')
    
    if len(clips.shape)==2:
        ax.plot(clips[:,0], clips[:,1],'b-', linewidth=2, c=cmap(0.1))
    else:
        map_idxs = list(np.arange(0,len(clips))/len(clips))
        for ic in range(clips.shape[0]):
            ax.plot(clips[ic,0,0], clips[ic,0,1], 'b-', linewidth=2, c=cmap(map_idxs[ic]))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return 

def plot_traj_lafan1(clips, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = cm.get_cmap('Spectral')
    
    if len(clips.shape)==2:
        ax.plot(clips[:,0], clips[:,2],'b-', linewidth=2, c=cmap(0.1))
    else:
        map_idxs = list(np.arange(0,len(clips))/len(clips))
        for ic in range(clips.shape[0]):
            ax.plot(clips[ic,0,0], clips[ic,0,2], 'b-', linewidth=2, c=cmap(map_idxs[ic]))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return 

def plot_lafan1(x, links, fps=30, save_path=None, colors = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    link_data = np.zeros((len(links),x.shape[0]-1,3,2))
    xini = x[0]

    if colors is None:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color='r')[0]
                    for st,ed in links]
    else:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color=colors[j])[0]
                    for j,(st,ed) in enumerate(links)]

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
    for i in range(1,x.shape[0]):
        for j,(st,ed) in enumerate(links):
            pt_st = x[i-1,st] #- y_rebase
            pt_ed = x[i-1,ed] #- y_rebase
            link_data[j,i-1,:,0] = pt_st
            link_data[j,i-1,:,1] = pt_ed
            
    def update_links(num, data_lst, obj_lst):
        cur_data_lst = data_lst[:,num,:,:] 
        cur_root = cur_data_lst[0,:,0]

        root_x = cur_root[0]
        root_y = cur_root[2]
        for obj, data in zip(obj_lst, cur_data_lst):
            obj.set_data(data[[0,2],:])
            obj.set_3d_properties(data[1,:])

            ax.set_xlim(root_x-1, root_x+1)
            ax.set_ylim(root_y-1, root_y+1)
            ax.set_zlim(0,2)
    
    line_ani = animation.FuncAnimation(fig, update_links, x.shape[0]-1, fargs=(link_data, link_obj),
                            interval=30, blit=False)
    if save_path is not None:
        writergif = animation.PillowWriter(fps=fps) 
        line_ani.save(save_path+'.gif', writer=writergif)

    if save_path is None:
        plt.show()
    plt.close()

def plot_style100(x, links, fps, save_path=None,  colors = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    link_data = np.zeros((len(links),x.shape[0]-1,3,2))
    xini = x[0]
    if colors is None:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color='r')[0]
                    for st,ed in links]
    else:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color=colors[j])[0]
                    for j,(st,ed) in enumerate(links)]


    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
    for i in range(1,x.shape[0]):
        for j,(st,ed) in enumerate(links):
            pt_st = x[i-1,st] #- y_rebase
            pt_ed = x[i-1,ed] #- y_rebase
            link_data[j,i-1,:,0] = pt_st
            link_data[j,i-1,:,1] = pt_ed
            
    def update_links(num, data_lst, obj_lst):
        cur_data_lst = data_lst[:,num,:,:] 
        cur_root = cur_data_lst[0,:,0]

        root_x = cur_root[0]
        root_y = cur_root[2]
        for obj, data in zip(obj_lst, cur_data_lst):
            obj.set_data(data[[0,2],:])
            obj.set_3d_properties(data[1,:])

            ax.set_xlim(root_x-1, root_x+1)
            ax.set_ylim(root_y-1, root_y+1)
            ax.set_zlim(0,2)
    
    line_ani = animation.FuncAnimation(fig, update_links, x.shape[0]-1, fargs=(link_data, link_obj),
                            interval=30, blit=False)
    if save_path is not None:
        writergif = animation.PillowWriter(fps=fps) 
        line_ani.save(save_path+'.gif', writer=writergif)

    if save_path is None:
        plt.show()
    plt.close()

def plot_amass(x, links, save_path=None, colors=None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    link_data = np.zeros((len(links),x.shape[0]-1,3,2))
    xini = x[0]
    if colors is None:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,1],xini[ed,1]],[xini[st,2],xini[ed,2]],color='r')[0]
                    for st,ed in links]
    else:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,1],xini[ed,1]],[xini[st,2],xini[ed,2]],color=colors[j])[0]
                    for j,(st,ed) in enumerate(links)]

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
    for i in range(1,x.shape[0]):
        for j,(st,ed) in enumerate(links):
            pt_st = x[i-1,st] #- y_rebase
            pt_ed = x[i-1,ed] #- y_rebase
            link_data[j,i-1,:,0] = pt_st
            link_data[j,i-1,:,1] = pt_ed
            
    def update_links(num, data_lst, obj_lst):
        cur_data_lst = data_lst[:,num,:,:] 
        cur_root = cur_data_lst[0,:,0]

        root_x = cur_root[0]
        root_y = cur_root[1]
        for obj, data in zip(obj_lst, cur_data_lst):
            obj.set_data(data[[0,1],:])
            obj.set_3d_properties(data[2,:])

            ax.set_xlim(root_x-1, root_x+1)
            ax.set_ylim(root_y-1, root_y+1)
            ax.set_zlim(0,2)
    
    line_ani = animation.FuncAnimation(fig, update_links, x.shape[0]-1, fargs=(link_data, link_obj),
                            interval=50, blit=False)
    if save_path is not None:
        writergif = animation.PillowWriter(fps=30) 
        line_ani.save(save_path+'.gif', writer=writergif)

    if save_path is None:
        plt.show()
    plt.close()


