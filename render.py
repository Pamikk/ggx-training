from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
dot_prod = lambda x,y: torch.sum(x*y, dim=-1,keepdim=True).clamp_min_(0.0)
from utils import generate_normalized_random_direction
class GGXRenderer(nn.Module):
    def __init__(self,multilight=True,albedo=0.0,roughness=0.7,fresnel=0.5) -> None:
        super(GGXRenderer,self).__init__()
        self.albedo = torch.tensor(albedo) 
        self.roughness = torch.tensor(roughness)
        self.fresnel = torch.tensor(fresnel)
        self.multilight = multilight
    def cal_F(self,VoH,fresnel):        
        FMi = ((-5.55473) * VoH - 6.98316) * VoH #interpolation
        return fresnel + (1.0-fresnel)*torch.pow(2.0,FMi)
    def cal_D(self,NoH,alpha):
        # specular D
        alpha2 = (alpha * alpha).clamp_min_(1e-6)
        nom0 =((NoH * alpha2 - NoH) * NoH + 1)# nom of D
        return alpha2/(nom0*nom0*np.pi) #(npt,nlights)
    def cal_G(self,NoL,NoV,alpha):
        k = (alpha/2.0)
        return 1/((NoV * (1 - k) + k) *(NoL * (1 - k) + k)).clamp_min_(1e-6)
    def cal_diffuse(self,albedo,fresnel):
        if len(fresnel.shape)>0 and(len(fresnel.shape)<len(albedo.shape)):
            fresnel = fresnel.unsqueeze(dim=-1).contiguous()
        return albedo/np.pi
    def forward(self,light_dir,view_dir,normal,albedo=None,roughness=None,fresnel=None):
        #all input should be flattened in spacial level 
        L = F.normalize(light_dir,dim=-1)#ray_num,light_num,3
        V = F.normalize(view_dir,dim=-1)
        albedo = self.albedo if albedo ==None else albedo
        roughness = self.roughness if roughness ==None else roughness
        fresnel = self.fresnel if fresnel == None else fresnel
        
        N = normal
        if self.multilight:
            assert len(L.shape)==3
            V = V.unsqueeze_(dim=1).contiguous()
            N = N.unsqueeze_(dim=1).contiguous()
            if len(albedo.shape)>0:
                albedo = albedo.unsqueeze_(dim=1).contiguous()
            if len(roughness.shape)>0:
                roughness = roughness.unsqueeze_(dim=1).contiguous()
            if len(fresnel.shape)>0:
                fresnel = fresnel.unsqueeze_(dim=1).contiguous()
        H = F.normalize((V+L)/2.0,dim=-1)
        alpha = roughness*roughness
        specular_D = self.cal_D(dot_prod(N,H),alpha)
        specular_G = self.cal_G(dot_prod(N,L),dot_prod(N,V),alpha)
        specular_F = self.cal_F(dot_prod(V,H),fresnel)
        spec = (specular_D*specular_G*specular_F)/4.0
        if len(spec.shape)<len(albedo.shape):
            spec = spec.unsqueeze_(dim=-1).contiguous()
        brdf = self.cal_diffuse(albedo,fresnel)+spec
        return brdf*(dot_prod(N,L)).float()


class GGXRenderer_optim(nn.Module):
    def __init__(self,multilight=False,h=512,device='cuda') -> None:
        super(GGXRenderer_optim,self).__init__()
        self.albedo = torch.rand((h,h,3),dtype=torch.float,device=device)
        self.roughness = torch.rand((h,h,1),dtype=torch.float,device=device)
        self.fresnel = torch.rand((h,h,3),dtype=torch.float,device=device)
        self.multilight = multilight
    def cal_F(self,VoH,fresnel):        
        FMi = ((-5.55473) * VoH - 6.98316) * VoH #interpolation
        return fresnel + (1.0-fresnel)*torch.pow(2.0,FMi)
    def cal_D(self,NoH,alpha):
        # specular D
        alpha2 = (alpha * alpha).clamp_min_(1e-6)
        nom0 =((NoH * alpha2 - NoH) * NoH + 1)# nom of D
        return alpha2/(nom0*nom0*np.pi) #(npt,nlights)
    def cal_G(self,NoL,NoV,alpha):
        k = (alpha/2.0)
        return 1/((NoV * (1 - k) + k) *(NoL * (1 - k) + k)).clamp_min_(1e-6)
    def cal_diffuse(self,albedo,fresnel):
        if len(fresnel.shape)>0 and(len(fresnel.shape)<len(albedo.shape)):
            fresnel = fresnel.unsqueeze(dim=-1).contiguous()
        return albedo/np.pi
    def forward(self,light_dir,view_dir,normal,hidx):
        #all input should be flattened in spacial level 
        L = F.normalize(light_dir,dim=-1)#ray_num,light_num,3
        V = F.normalize(view_dir,dim=-1)
        albedo = self.albedo[hidx,:,:]
        roughness = self.roughness[hidx,:,:]
        fresnel = self.fresnel[hidx,:,:]
        
        N = normal
        if self.multilight:
            assert len(L.shape)==3
            V = V.unsqueeze_(dim=1).contiguous()
            N = N.unsqueeze_(dim=1).contiguous()
            if len(albedo.shape)>0:
                albedo = albedo.unsqueeze_(dim=1).contiguous()
            if len(roughness.shape)>0:
                roughness = roughness.unsqueeze_(dim=1).contiguous()
            if len(fresnel.shape)>0:
                fresnel = fresnel.unsqueeze_(dim=1).contiguous()
        H = F.normalize((V+L)/2.0,dim=-1)
        alpha = roughness*roughness
        specular_D = self.cal_D(dot_prod(N,H),alpha)
        specular_G = self.cal_G(dot_prod(N,L),dot_prod(N,V),alpha)
        specular_F = self.cal_F(dot_prod(V,H),fresnel)
        spec = (specular_D*specular_G*specular_F)/4.0
        if len(spec.shape)<len(albedo.shape):
            spec = spec.unsqueeze_(dim=-1).contiguous()
        brdf = self.cal_diffuse(albedo,fresnel)+spec
        return brdf*(dot_prod(N,L)).float()

def get_rendering(wi,wo,data,Render,diffuse=True,light_intensity = 3.0):
    if data.shape[-1] ==12:
        normal,albedo,roughness,fresnel = data.split([3,3,3,3],dim=-1)
    if data.shape[-1] ==10:
        normal,albedo,roughness,fresnel = data.split([3,3,1,3],dim=-1)
    if (data.shape[-1]==9):
        normal,albedo,fresnel = data.split([3,3,3],dim=-1)
        roughness = None
    if (data.shape[-1] ==7) and diffuse:
        normal,albedo,roughness= data.split([3,3,1],dim=-1)
        fresnel = None
    if (data.shape[-1] ==7) and (not diffuse):
        normal,roughness,fresnel= data.split([3,1,3],dim=-1)
        albedo = None
    if (data.shape[-1] == 6):
        normal,albedo = data.split([3,1,3],dim=-1)
        roughness = fresnel = None
    img = (light_intensity* np.pi*Render(wi,wo,normal,albedo,roughness,fresnel))
    img = img.clamp_(1e-10,1.0) 
    return img



