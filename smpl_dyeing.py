import numpy as np
import os 
import torch
import random
import pywavefront
from tqdm import tqdm
import trimesh
import math
from matplotlib import cm
from matplotlib.colors import Normalize

color_map = [
    [1, 0, 0, 1],  # 红色
    [1, 0.5, 0, 1],  # 橙色
    [1, 1, 0, 1],  # 黄色
    [0.5, 1, 0, 1],  # 黄绿色
    [0, 1, 0, 1],  # 绿色
    [0, 1, 0.5, 1],  # 青绿色
    [0, 1, 1, 1],  # 青色
    [0, 0.5, 1, 1],  # 蓝青色
    [0, 0, 1, 1],  # 蓝色
    [0.5, 0, 1, 1]  # 紫色
]

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

def sigmoid(z):
    return 1/(1+math.exp(-z))

def dyeing(base_dir, objnames):
    objs, faces = obj_load(base_dir, objnames)
    cut = lambda x: x[:3]
    for objname, vertices, face in zip(objnames, objs, faces):
        colors = np.zeros((len(vertices), 4))
        vertices = list(map(cut, vertices))
        z_max = np.array(vertices)[2].max()
        
        for i, vertex in enumerate(vertices):
            z_value = vertex[2]
            colors[i] = cm.jet(1-z_value/z_max)
            # if z_value > 0.3:
            #     colors[i] = [1,1,1,1]

        mesh = trimesh.Trimesh(vertices=vertices, faces=face, vertex_colors=colors)

        mesh.export(f'./output/{objname}.ply')
        
        
        
if __name__=='__main__':
    base_dir = './data'
    names = ['09417']
    dyeing(base_dir, names)
    
    
    