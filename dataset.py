import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from render import get_rendering
from torch.nn import functional as F
path = '/home/pami/dataset/materialsData_multi_image'
def get_id(fname):
    return int(fname.split('_')[0])
def process_to_flat_tensor(data):
    return torch.tensor(data,dtype=torch.float32)
import math
def gen_normalized_dir(r,phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(1.0 - np.square(r))
    finalVec = np.stack([x, y, z], axis=-1) #(3,)
    return finalVec
def generate_normalized_random_direction(lowEps = 0.001, highEps =0.05):
    r1 = np.random.uniform(0.0 + lowEps, 1.0 - highEps)
    r2 =  np.random.uniform( 0.0, 1.0)
    r = np.sqrt(r1)
    phi = 2 * math.pi * r2
       
    return torch.tensor(gen_normalized_dir(r,phi))
from render import GGXRenderer
mutilight = False
Render = GGXRenderer(multilight=mutilight)
def gen_DiffuseRendering(data,Render):
    currentViewPos = generate_normalized_random_direction()
    currentLightPos = generate_normalized_random_direction()
    
    wi = currentLightPos
    wi = wi.unsqueeze_(dim=0)
    
    wo = currentViewPos
    wo = wo.unsqueeze_(dim=0)
    if mutilight:
       wi = wi.unsqueeze_(dim=0)
    length = data.shape[0]
    img = torch.zeros((data.shape[0],3),dtype=data.dtype)
    for idx in range(0,length,1024):
        img[idx:idx+1024,:] = get_rendering(wi,wo,data[idx:idx+1024,:],Render)
    return (wi,wo),img
def generate_distance():
    gaussian =np.random.normal(0.5, 0.75)
    return (np.exp(gaussian))
def gen_SpecularRendering(data,Render):    

    currentViewDir = generate_normalized_random_direction()
    currentLightDir = currentViewDir * torch.tensor([-1.0, -1.0, 1.0])
    #Shift position to have highlight elsewhere than in the center.
    currentShift = torch.concat([torch.rand(2)*2-1,torch.zeros(1) + 0.0001], axis=-1)
    
    currentViewPos = currentViewDir + currentShift
    currentLightPos =currentLightDir*generate_distance() + currentShift
    
    length = data.shape[0]
    h = int(np.sqrt(length))
    XsurfaceArray = torch.linspace(-1.0, 1.0, h).unsqueeze(1)
    XsurfaceArray = XsurfaceArray.repeat(1,h)
    YsurfaceArray = -1 * torch.transpose(XsurfaceArray,0,1) #put -1 in the bottom of the table

    surfaceArray = torch.stack([XsurfaceArray, YsurfaceArray, torch.zeros([h, h])],dim=-1)
    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray
    if mutilight:
       wi = wi.unsqueeze_(dim=0)
    
    img = torch.zeros((data.shape[0],3),dtype=data.dtype)
    for idx in range(0,length,1024):
        img[idx:idx+1024,:] = get_rendering(wi.view(-1,3)[idx:idx+1024,:],wo.view(-1,3)[idx:idx+1024,:],data[idx:idx+1024,:],Render)
    return (wi,wo),img
tosave = ['psnr','rre']
class Material(Dataset):
    def __init__(self,path=path,mode='train',tsize=(512,512),width=256,multilight = True,overfitting=False) -> None:
        super(Material,self).__init__()
        if mode == 'trainval':
            split = 'train'
        else:
            split = mode
        self.imgs = sorted([fn for fn in os.listdir(os.path.join(path,split))],key=lambda x:get_id(x))
        if overfitting:
            self.imgs = self.imgs[:1]
        self.ovf = overfitting
        self.tsize = tsize
        self.mode = mode
        self.fpath = path
        self.width = width
        self.split = split
        self.multilight =multilight
    def __len__(self):
        if self.mode=='trainval':
           return len(self.imgs[:min(10,len(self.imgs))])
        else:
            return len(self.imgs)
    def process_normal(self,normal):
        normal_image = np.array(normal)[:, :, :3]  # (-1.0, 1.0) * 0.5 + 0.5
        normal_direction = (normal_image - 0.5) * 2.0
        return normal_direction

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.fpath,self.split,self.imgs[index]))
        img = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).astype(float)/255.0
        
        h,w,_ = img.shape
        downsample = self.width*1.0/(h*1.0)
        img= cv2.resize(img,(int(downsample*w),self.width))
        h,w,_ = img.shape
        assert w == 4*self.width
        normal,albedo,roughness,fresnel = np.split(img,4,axis=1)
        normal = F.normalize(torch.tensor(self.process_normal(normal),dtype=torch.float32),dim=-1)
        albedo = torch.tensor(albedo,dtype=torch.float32)
        roughness = torch.tensor(roughness,dtype=torch.float32)
        fresnel = torch.tensor(fresnel,dtype=torch.float32)

        data = torch.cat([normal,albedo,roughness,fresnel],dim=-1)
        #fix light if overf by fixing seeds
        if self.ovf:
            np.random.seed(1234)
        (wi,wo),img = gen_DiffuseRendering(data.view(-1,12),GGXRenderer(self.multilight))
        #(wi_s,wo_s),img_s = gen_SpecularRendering(data.view(-1,12),GGXRenderer(self.multilight))
        return wi,wo,img.reshape(h,self.width,3),data

class MaterialImg(Dataset):
    def __init__(self,path=path,mode='train',tsize=(512,512),width=512) -> None:
        super(Material,self).__init__()
        if mode == 'trainval':
            split = 'train'
        else:
            split = mode
        self.imgs = sorted([fn for fn in os.listdir(os.path.join(path,split))],key=lambda x:get_id(x))[:20]
        self.tsize = tsize
        self.mode = mode
        self.fpath = path
        self.width = 512
        self.split = split
    def __len__(self):
        if self.mode=='trainval':
           return len(self.img_paths[:20])
    def process_normal(self,normal):
        normal_image = np.array(normal)[:, :, :3]  # (-1.0, 1.0) * 0.5 + 0.5
        normal_direction = (normal_image - 0.5) * 2.0
        return normal_direction

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.fpath,self.split,self.imgs[index]))
        img = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).astype(float)/255.0
        h,w,_ = img.shape
        assert w == 4*self.width
        normal,albedo,roughness,fresnel = np.split(img,4)
        normal = self.process_normal(normal)

        return tuple(map(process_to_flat_tensor,(normal,albedo,roughness,fresnel)))
