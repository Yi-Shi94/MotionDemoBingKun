from PIL import Image,ImageDraw,ImageFont
from utils.sampling import FPS
import numpy as np
import matplotlib.pyplot as plt

im_w = 500
im_h = 200
rw = 20 

def text2png(img_path, font_path, text="SHAILAB"):     
    img = Image.new('RGB', (im_w, im_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 18)
    text = text

    #(w,h) = draw.textbbox((0,0),text=text, font=font)
    #x = (img.width - w) / 2
    #y = (img.height - h) / 2

    # Draw the text on the image
    draw.text((0, 0), text, (0, 0, 0), font=font)
    img.save(img_path)
    

def png2coord(img_path):
    
    with Image.open(img_path) as im:
        pixels = list(im.getdata())
    
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    coord = []
    rw_w = rw/im_w
    rw_h = -rw/im_h

    for i in range(len(pixels)):
        for j in range(len(pixels[0])):
            if pixels[i][j][0]<10 and pixels[i][j][1]<10 and pixels[i][j][2]<10 :
                coord.append((rw_w*j*3,rw_h*i))
   
    coord = np.array(coord)
    coord_center = np.mean(coord, axis=0)
    coord -= coord_center
    return coord

def coord2pc(coord, n_samples):
    N = coord.shape[0]
    print('InTotal:{} points'.format(N))
    
    if N < n_samples:
        pass
    else:
        fps = FPS(coord, n_samples=n_samples)
        coord = fps.fit()
    
    print('Sampled:{} points'.format(coord.shape[0]))
    return coord        


def st_point_gen(coord, radius=10, mode='random'):
    N = coord.shape[0]
    if mode == 'random':
        st_coord = np.random.randn(N,2) * radius
    
    elif mode == 'circle':
        thetas = np.arange(N)*(2*np.pi)/N
        xs = radius*np.cos(thetas)
        ys = radius*np.sin(thetas)
        st_coord = np.array([xs,ys]).transpose()
    else:
        raise NotImplementedError 

    return st_coord
    

def plot_scatter_multi(coord, st_coord, path):
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    x,y = list(zip(*coord))
    plt.scatter(x, y, label='test', color='k', s=25, marker="o")
    x,y = list(zip(*st_coord))
    plt.scatter(x, y, label='test', color='b', s=25, marker="o")
    plt.savefig(path)


def plot_scatter(coord, path):
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    x,y = list(zip(*coord))
    plt.scatter(x, y, label='test', color='k', s=25, marker="o")
    plt.savefig(path)


if __name__=='__main__':
    text = 'SHAILAB'
    font_path = 'fonts/arial.ttf'
    img_save_path = 'results/font_img.png'
    pc_save_path = 'results/pc.npz'

    text2png(img_save_path, font_path, text)
    coord = png2coord(img_save_path)
    plot_scatter(coord,'results/pc_ori.png')
    
    coord = coord2pc(coord, n_samples=100)
    plot_scatter(coord,'results/pc_samp.png')

    st_coord = st_point_gen(coord, radius=2.8, mode='circle')
    plot_scatter_multi(coord, st_coord, 'results/pc_st.png')

    np.savez(pc_save_path,coord=coord,st_coord=st_coord)