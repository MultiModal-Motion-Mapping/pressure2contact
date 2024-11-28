import numpy as np
import os
import smplx
import torch
# import articulate as art
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端, 避免和cv2冲突
import matplotlib.pyplot as plt
import time
import pickle
import cv2
import glob
from tqdm import tqdm


def get_file_list(path):
    file_names = os.listdir(path)
    return [f for f in file_names if os.path.isfile(os.path.join(path, f))]


def get_folder_list(path):
    all_items = os.listdir(path)
    return [item for item in all_items if os.path.isdir(os.path.join(path, item))]


class SMPL_joint_set:
    joint_names = [
    "pelvis",                   # 0 
    "left_hip",               # 1 (0,1,2)
    "right_hip",            # 2 (3,4,5)
    "spine1",                 # 3 (6,7,8)
    "left_knee",            # 4 (9,10,11)
    "right_knee",         # 5 (12,13,14)
    "spine2",                  # 6 (15,16,17)
    "left_ankle",          # 7 (18,19,20)
    "right_ankle",        # 8 (21,22,23) 
    "spine3",                 # 9 (24,25,26)
    "left_foot",             # 10 (27,28,29)
    "right_foot",          # 11 (30,31,32)
    "neck",                     # 12 (33,34,35)
    "left_collar",          # 13 (36,37,38)
    "right_collar",       # 14 (39,40,41)
    "head",                     # 15 (42,43,44)
    "left_shoulder",    # 16 (45,46,47)
    "right_shoulder", # 17 (48,49,50)
    "left_elbow",          # 18 (51,52,53)
    "right_elbow",       # 19 (54,55,56)
    "left_wrist",            # 20 (57,58,59)
    "right_wrist",         # 21 (60,61,62)
    "left_hand",           # 22 (63,64,65)
    "right_hand"         # 23 (66,67,68)
    ]
    joint_graph = [[0, 3, 6,   9, 12, 0, 1, 4,    7, 0, 2, 5,   8,    9, 13, 16, 18, 20,   9, 14, 17, 19, 21],
                                  [3, 6, 9, 12, 15, 1, 4, 7, 10, 2, 5, 8, 11, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]]
    carpet_graph = [[0, 0, 1, 2], 
                                    [1, 2, 3, 3]]
    parent = joint_graph[0] +carpet_graph[0]
    son = joint_graph[1] + carpet_graph[1]
    whole_graph = [parent, son]


def save_as_npy(data, path):
    np.save(path, data)
    
def compute_contact_region_single(point):
        # 确保 point1 和 point2 是整数元组
    point = tuple(map(int, point))

    # print("Image shape:", image_array.shape, "dtype:", image_array.dtype)  # 打印输入图像的信息，便于调试
    RADIUS = 3

    x_min = point[0] - RADIUS
    y_min = point[1] - RADIUS
    x_max = point[0] + RADIUS
    y_max = point[1] + RADIUS

    # 计算 bounding box 的位置和大小
    # bounding_box = {
    #     "top_left": (x_min, y_min),
    #     "bottom_right": (x_max, y_max),
    #     "width": x_max - x_min,
    #     "height": y_max - y_min
    # }
    # x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
    
    return x_min, x_max, y_min, y_max

def compute_contact_region(point1, point2):
    """
    :param image_array: np.array, shape = (y, x) = (160, 120)
    :return: 
    """
    # 选择一个适当的范围，指定这两个点的邻域，比如半径为 10 像素
    # radius = 15

    # 确保 point1 和 point2 是整数元组
    point1 = tuple(map(int, point1))
    point2 = tuple(map(int, point2))

    # print("Image shape:", image_array.shape, "dtype:", image_array.dtype)  # 打印输入图像的信息，便于调试

    x_min = min(point1[0], point2[0]) - 5
    y_min = min(point1[1], point2[1]) - 8
    x_max = max(point1[0], point2[0]) + 5
    y_max = max(point1[1], point2[1]) + 5

    # 计算 bounding box 的位置和大小
    bounding_box = {
        "top_left": (x_min, y_min),
        "bottom_right": (x_max, y_max),
        "width": x_max - x_min,
        "height": y_max - y_min
    }
    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
    
    return x_min, x_max, y_min, y_max
    

def save_figure(f, color_image, pressure_dir):
    if not os.path.exists(pressure_dir):
        os.makedirs(pressure_dir)

    # 保存带有 bounding box 的图片
    filename = f"frame_{f}_contact_region.jpg"
    save_path = os.path.join(pressure_dir, filename)
    cv2.imwrite(save_path, color_image)
    print(f"图片已保存到: {save_path}")



def compute_point_with_min_distance(point, points):
    """
    :param point: np.array, shape = (3,)
    :param points: np.array, shape = (x, y, 3)
    :return: 
    """
    distances = np.linalg.norm(points - point, axis=2)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return min_idx




def compute_pressure_pos(pressure, smpl, path, save_fig=False):
    """
    :param pressure: np.array, shape = (num_frame, y, x) = (num_frame, 160, 120)
    :param smpl: smplx model
    """
    
    print("Pressure shape:", pressure.shape) 
    scale = 2.0/pressure.shape[1]
    carpet_pos = np.zeros((pressure.shape[2], pressure.shape[1], 3))
    for i in range(pressure.shape[2]):
        for j in range(pressure.shape[1]):
            carpet_pos[i, j, 0] = i*scale + 0.5*scale
            carpet_pos[i, j, 1] = -j*scale - 0.5*scale
            carpet_pos[i, j, 2] = 0.
    assert carpet_pos.shape == (120, 160, 3)
        
    
    joint_pos = smpl.joints[:, :24, :].cpu().numpy()
    assert joint_pos.shape[1] == 24 and joint_pos.shape[2] == 3

        
    contact = np.zeros((pressure.shape[0], 24))

    for f in tqdm(range(pressure.shape[0])):
        """
        joint pos and carpet pos are in the same coordinate system
        pressure array is in the different coordinate system
        """
        pressure_ = pressure[f, :, :]
        # do flip to make the image in the right direction
        pressure_ = cv2.flip(pressure_, 1)
        pressure_ = cv2.flip(pressure_, 0)
        joint_pos_ = joint_pos[f, :, :]
        # Filter z-height
        
        for idx in range(24):
            if joint_pos[f, idx, 2] >= 0.1 :
                continue
            carpet_idx = compute_point_with_min_distance(joint_pos_[idx, :], carpet_pos)
            lxmin_a, lxmax_a, lymin_a, lymax_a = compute_contact_region_single(carpet_idx)
            # 计算该区域内的像素之和
            carpet_pressure = pressure_[lymin_a:lymax_a, lxmin_a:lxmax_a]
            contact_sum = np.sum(carpet_pressure)
            if contact_sum > 100:
                contact[f, idx] = 1
    
    contact = contact[:, [1,2,4,5,7,8,10,11,20,21,22,23]]
    
    save_path = os.path.join(path, 'contact.npy')
    print('Contact shape {%s}, Save path {%s}'%(contact.shape, save_path))
    np.save(save_path, contact)



def compute_smpl_model(path):
    device = torch.device('cpu')
    smpl_model_path = './SMPL_NEUTRAL.pkl'


    smpl_name = os.path.join(path, 'smpl.npy')
    pressure_name = os.path.join(path, 'pressure.npz')
    # print("Processing {} ...".format(smpl_name))
    smpl_data = np.load(smpl_name, allow_pickle=True).item()
    pressure = np.load(pressure_name)["pressure"]
    
    body_pose = torch.Tensor(smpl_data['body_pose']).reshape(-1, 23, 3).to(device)  # >>> torch.Size([17442, 23, 3])
    num_frames = body_pose.shape[0]  # >>>  17442
    betas = torch.Tensor(smpl_data['betas']).expand(num_frames, -1).to(device)  # >>> torch.Size([17442, 10])
    global_orient = torch.Tensor(smpl_data['global_orient']).unsqueeze(1).to(device)  # >>>  torch.Size([17442, 1, 3])
    transl = torch.Tensor(smpl_data['transl']).to(device)  # >>>  torch.Size([17442, 3])
    smpl_create = smplx.create(smpl_model_path).to(device)
    smpl_model = smpl_create(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    

    compute_pressure_pos(pressure, smpl_model, path)  

def gen_contact():
    base_dir = '/nasdata/shenghao/Pressure_release'
    for root, dirs, files in os.walk(base_dir, topdown=False):
        if 'keypoints.npy' in files and 'pressure.npz' in files and 'smpl.npy' in files:
            print("Now begin to process {%s}"%root)
            compute_smpl_model(root)
            
    
    
    
    


    
def nasdata2data1(base_dir):
    gt_files_smpl = findAllFilesWithSpecifiedName(base_dir, 'smpl.npy')
    gt_files_smpl.sort()

    for gt_file_smpl in gt_files_smpl:
        print(gt_file_smpl)
        gt_file_new_smpl = gt_file_smpl.replace('/nasdata/', '/data1/')
        gt_file_new_kps = gt_file_new_smpl.replace('smpl.npy', 'keypoints.npy')
        gt_file_new_pressure = gt_file_new_smpl.replace('smpl.npy', 'pressure.npz')
        gt_file_new_contact = gt_file_new_smpl.replace('smpl.npy', 'contact.npy')
        gt_file_new_video = gt_file_new_smpl.replace('smpl.npy', '1.mp4')
        print(gt_file_new_smpl)
        gt_new_dir = os.path.dirname(gt_file_new_smpl)
        if not os.path.exists(gt_new_dir):
            os.makedirs(gt_new_dir)
        os.system('cp ' + gt_file_smpl + ' ' + gt_file_new_smpl)
        os.system('cp ' + gt_file_smpl.replace('smpl.npy', 'keypoints.npy') + ' ' + gt_file_new_kps)
        os.system('cp ' + gt_file_smpl.replace('smpl.npy', 'pressure.npz') + ' ' + gt_file_new_pressure)
        os.system('cp ' + gt_file_smpl.replace('smpl.npy', 'contact.npy') + ' ' + gt_file_new_contact)
        os.system('cp ' + gt_file_smpl.replace('smpl.npy', '1.mp4') + ' ' + gt_file_new_video)

if __name__ == '__main__':
    base_dir = '/nasdata/shenghao/Pressure_release/0729/qnx/2024-07-29-17-19-22'
    print("Processing {} ...".format(base_dir))
    gen_contact()
    



    

        