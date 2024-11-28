import numpy as np
import os
import torch
import glob
import math
import os.path as osp
import smplx
import trimesh
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

smpl_left_leg = [0,1,4,7,10]
smpl_right_leg = [0,2,5,8,11]
smpl_left_arm = [9,13,16,18,20]
smpl_right_arm = [9,14,17,19,21]
smpl_head = [9,12,15]
smpl_body = [9,6,3,0]

def split_kps():
    dict_file = "data/ours/resnet18_small_gru2_4resbi_contact1_wholebody_10to1/5e-05/eval_result_latest.npy"
    dict = np.load(dict_file, allow_pickle=True).item()['joint']
    pressure_kps = dict.reshape(-1, 22, 3)
    print(pressure_kps.shape)

    gt_files = glob.glob("data/gt/*.npy")
    print(len(gt_files))

    index_old = 0

    for gt_file in gt_files:
        gt_kps = np.load(gt_file)[:,:22]
        print(gt_kps.shape)

        temp = math.ceil(gt_kps.shape[0] / 20)
        index_old = temp * 20 + index_old
        pred_kps = pressure_kps[index_old - temp*20:index_old]
        pred_kps = pred_kps[:gt_kps.shape[0]]
        print(pred_kps.shape)

        name = gt_file.split("\\")[-1].split(".")[0]
        output_dir = osp.join(dict_file.split(".")[0], name)
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        np.save(osp.join(output_dir, "pred_kps.npy"), pred_kps)

def plot_keypoints():
    keypoints_file = "/nasdata/shenghao/Pressure_release/0729/qnx/2024-07-29-17-19-22/keypoints.npy"
    kps = np.load(keypoints_file, allow_pickle=True)
    kps = kps.reshape(-1, 24, 3)

    contact_file = './contact.npy'
    contact = np.load(contact_file, allow_pickle=True)

    output_dir = './Data'
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i in tqdm(range(0, kps.shape[0])):
        contact_label_ = (contact[i]==0)
        contact_label = np.nonzero(contact[i])
        
        ax.set_xlim3d([0, 1.5])
        ax.set_zlim3d([0, 2])
        ax.set_ylim3d([-2, 0])
        ax.scatter3D(kps[i,contact_label_,0], kps[i,contact_label_,1], kps[i,contact_label_,2], color='green', s=20)
        ax.scatter3D(kps[i,contact_label,0], kps[i,contact_label,1], kps[i,contact_label,2], color='black', s=40)
        
        ax.plot3D(kps[i,smpl_left_leg,0], kps[i,smpl_left_leg,1], kps[i,smpl_left_leg,2], 'red')
        ax.plot3D(kps[i,smpl_left_arm,0], kps[i,smpl_left_arm,1], kps[i,smpl_left_arm,2], 'red')
        ax.plot3D(kps[i,smpl_right_leg,0], kps[i,smpl_right_leg,1], kps[i,smpl_right_leg,2], 'red')
        ax.plot3D(kps[i,smpl_right_arm,0], kps[i,smpl_right_arm,1], kps[i,smpl_right_arm,2], 'red')
        ax.plot3D(kps[i,smpl_head,0], kps[i,smpl_head,1], kps[i,smpl_head,2], 'red')
        ax.plot3D(kps[i,smpl_body,0], kps[i,smpl_body,1], kps[i,smpl_body,2], 'red')
        # plt.show()
        plt.savefig(output_dir +"/" +str(i).zfill(5)+".jpg")

        ax.clear()



def render_smpl():
    dict_file = "data/imgpressure2pose/gbtc_resnet_smallconv1_contact5_2d1_scretch_test/5e-05/eval_result_latest.npy"
    # dict_file = "data/ours/resnet18_small_gru4resbi_contact1_wholebody_5to1/5e-05/eval_result_latest/2024-07-29-10-43-02/pred_kps.npy"
    smpl_params = np.load(dict_file, allow_pickle=True).item()['smpl_params']
    print(smpl_params.shape)
    smpl_params = torch.from_numpy(smpl_params)
    pred_beta, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
    pred_global_orient, pred_body_pose = torch.split(pred_theta, [3, 69], dim=1)

    smpl = smplx.create('smpl/SMPL_NEUTRAL.pkl')

    smpl_result = smpl(betas=pred_beta, # shape parameters
                    body_pose=pred_body_pose, # pose parameters
                    global_orient=pred_global_orient, # global orientation
                    transl=pred_trans) # global translation
    
    vertices = smpl_result.vertices.detach().cpu().numpy()

    output_dir = dict_file.split(".")[0]
    output_dir = osp.join(output_dir, "smpl")
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(2672, 2673):
        mesh = trimesh.Trimesh(vertices=vertices[i], faces=smpl.faces)
        mesh.export(output_dir +"/" +str(i).zfill(5)+".obj")

def plot_heatmap(data):
    colors = ['Reds','PuRd', 'Oranges', 'YlOrRd', 'YlOrBr','Greens','BuGn', 'YlGn',
              'PuRd', 'GnBu', 'YlGnBu', 'Blues', 'YlOrRd', 'YlOrBr', 'Greens', 'GnBu', 'YlGnBu', 'Blues','Greens','BuGn', 'YlGn','Reds']

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))

    # ax.set_xlim(0,26)
    # ax.set_ylim(0,18)
    # ax.set_zlim(0,36)

    # plt.xticks([0,5,10,15])
    # plt.yticks([0,5,10,15])
    # ax.set_zticks([0,5,10,15])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.view_init(210,230)

    # for i in range(data.shape[0]):
    for i in range(22):
        frame = np.reshape(data[i,:,:,:],(26,36,36))
        # print(frame.shape)
        flag = frame > 0
        x,y,z = np.where(frame>0.05)
        ax.scatter(x, y, z, c=frame[x,y,z]*255, cmap=colors[i])
        # ax.scatter(x, y, z,  cmap=colors[i])

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    plt.imsave('heatmap.png', img)
if __name__ == "__main__":
    # heatmap_dir = "/data1/shenghao/Pressure_release/0729/csy/2024-07-29-09-58-38/heatmap3D/heatmap3D_01000.npy"
    # heatmap = np.load(heatmap_dir)
    # print(heatmap.shape)
    # plot_heatmap(heatmap)

    plot_keypoints()

    # split_kps()