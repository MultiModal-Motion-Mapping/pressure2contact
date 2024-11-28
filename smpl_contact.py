import numpy as np
import smplx
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import cv2
import random
import pywavefront
import trimesh
import math
from matplotlib import cm
import matplotlib.pyplot as plt

def save_ply_with_color(filename, vertices, faces, colors):
    with open(filename, 'w') as ply_file:
        # 写入 PLY 文件头部信息
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")
        # 写入带颜色的顶点
        for v, color in zip(vertices, colors):
            r, g, b = (color * 255).astype(int)
            ply_file.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b}\n")
        # 写入面片信息
        for f in faces:
            ply_file.write(f"3 {f[0]} {f[1]} {f[2]}\n")

def obj_load(base_dir, objnames):
    objs = []
    faces = []
    for objname in objnames:
        # 加载 .obj 文件
        obj_path = os.path.join(base_dir, objname + '.obj')
        wavefront_obj = pywavefront.Wavefront(obj_path, collect_faces=True)
        # 提取顶点和面
        objs.append(wavefront_obj.vertices)
        faces.append(wavefront_obj.mesh_list[0].faces if wavefront_obj.mesh_list else [])
    return objs, faces
      
def smpl_load(base_dir):
    data = np.load(os.path.join(base_dir, 'smpl.npy'), allow_pickle=True).item()
    global_orient = data['global_orient']
    num_frames = global_orient.shape[0]
    global_orient = torch.from_numpy(global_orient).unsqueeze(-2).float()
    betas = torch.from_numpy(data['betas']).expand(num_frames, -1).float()
    body_pose = torch.from_numpy(data['body_pose']).reshape(-1, 23, 3).float()
    transl = torch.from_numpy(data['transl']).float()
    smpl_model_path = './SMPL_NEUTRAL.pkl'
    smpl_create = smplx.create(smpl_model_path)
    result = smpl_create(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    print('Now you have downloaded smpl!')
    return result, smpl_create

def getlabels(contact, vertices):
    labels = []
    for i in tqdm(range(len(vertices))):
        vertice = vertices[i]
        x, y, z = vertice[0], vertice[1], vertice[2]
        if contact[y,x] == 1 and z < 0.03:
            if len(labels) == 0:
                labels.append([x,y,z,i])
                continue
            labels.append([x,y,z,i])
            # for label in labels[:-1]:
            #     if label[0] == x and label[1] == y:
            #         if label[2] < z:
            #             labels.remove([x,y,z,i])
            #             break
            #         else:
            #             labels.remove(label)
            #             break
    labels = [item[3] for item in labels]
    return labels

def equalize(lista):
    hist, bins = np.histogram(lista, bins=np.arange(257)) 
    cdf = np.cumsum(hist) 
    cdf_normalized = cdf / cdf[-1] 
    equalized_data = np.interp(lista, bins[:-1], cdf_normalized).astype(float)
    return equalized_data

def dyeing(press, vertices):
    f = lambda x: (x-35)/(203-35)
    colors = np.ones((len(vertices), 3))
    for i in tqdm(range(len(vertices))):
        vertex = vertices[i]
        x, y, z = vertex[0], vertex[1], vertex[2]
        press_ = press[y,x]
        if press_ > 0 and z < 0.03:
            colors[i] = cm.Spectral(f(x))[:3]
            print(f(x))
    return colors


def press_load(base_dir):
    data = np.load(os.path.join(base_dir, 'pressure.npz'), allow_pickle=True)['pressure']
    print("Now you have downloaded press!")
    return data

def contact_load(base_dir):
    data = np.load(os.path.join(base_dir, 'pressure.npz'), allow_pickle=True)['pressure']
    contact = np.where(data > 10, 1, 0)
    print("Now you have downloaded contact!")
    return contact

def smpl_contact(base_dir):
    result, smpl_create = smpl_load(base_dir)
    index = 1000
    vertices = result.vertices[index].numpy()  
    faces = smpl_create.faces  

    contact_label = torch.nonzero(result.vertices[index, :, -1] < 0).squeeze().numpy()
    colors = np.ones((vertices.shape[0], 3))  
    colors[contact_label] = [1, 0, 0]  

    # 导出为 PLY 文件
    save_ply_with_color('./output.ply', vertices, faces, colors)
    print('Now you have output one ply file!')

def z_tosmpl(base_dir, objnames):
    objs, faces = obj_load(base_dir, objnames)
    cut = lambda x: x[:3]
    for objname, vertices, face in zip(objnames, objs, faces):
        vertices = list(map(cut, vertices))
        colors = np.ones((len(vertices), 3)) 
        for i, vertex in enumerate(vertices):
            z_value = vertex[2]
            if z_value < 0.02:
                colors[i] = [1,0,0]
                
        save_ply_with_color(f'./output/{objname}.ply', vertices, faces[0], colors)
        print(f"Now you have made {objname}!")
    
    
def contact_tosmpl(base_dir, objnames):
    # result, smpl_create = smpl_load(base_dir)
    contact = contact_load(base_dir)
    # press = press_load(base_dir)
    objs, faces = obj_load(base_dir, objnames)
    
    project = lambda x: [int(x[0]*80), -int(x[1]*80), float(x[2])]
    
    vertices = objs[0]
    contact_ = contact[9417]
    contact_ = cv2.flip(contact_, 0)
    contact_ = cv2.flip(contact_, 1)
    vertices_contact = map(project, vertices)
    vertices_contact = list(vertices_contact)
    # colors = dyeing(contact_, vertices_contact)
    colors = np.ones((len(vertices), 3))
    labels = getlabels(contact_, vertices_contact)
    colors[labels] = [105/255, 225/255, 151/255]
    
    save_ply_with_color(f'./output/{objnames[0]}.ply', vertices, faces[0], colors)
    print("Now you have made a new ply file!")
    
    # choices = np.random.choice(range(contact.shape[0]), 100, replace=False)
    # for index in tqdm(choices):
    #     vertices = result.vertices[index].numpy()
    #     contact_ = contact[index]
    #     contact_ = cv2.flip(contact_, 0)
    #     contact_ = cv2.flip(contact_, 1)
        
    #     # project
    #     vertices_contact = map(project, vertices[:,0].reshape(-1, 1), vertices[:,1].reshape(-1,1), vertices[:,2].reshape(-1,1))
    #     vertices_contact = list(vertices_contact)
        
    #     # Get contact_labels
    #     contact_labels = getlabels(contact=contact_, vertices=vertices_contact)
        
    #     faces = smpl_create.faces  
    #     colors = np.ones((vertices.shape[0], 3))  
    #     colors[contact_labels] = [1, 0, 0]  
    #     save_ply_with_color('./output/output%d.ply'%index, vertices, faces, colors)
    # print('Now you have finished output!')
    
    
if __name__=='__main__':
    base_dir = '/nasdata/shenghao/Pressure_release/0729/csy/2024-07-29-09-58-38'
    names = ['09417']
    # z_tosmpl('./data', names)
    contact_tosmpl('./data', names)