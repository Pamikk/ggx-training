from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
dot_prod = lambda x,y: torch.sum(x*y, dim=-1,keepdim=True).clamp_min_(0.0)
from utils import generate_normalized_random_direction
light_intensity = 2.5
class GGXRenderer(nn.Module):
    def __init__(self,multilight=True,albedo=0.0,roughness=0.7,fresnel=0.5) -> None:
        super(GGXRenderer,self).__init__()
        self.albedo = torch.tensor(albedo) 
        self.roughness = torch.tensor(roughness)
        self.fresnel = torch.tensor(fresnel)
        self.normal = torch.rand((512,512,3),dtype=torch.float)
        self.hidx = None
        self.multilight = multilight
    def cal_F(self,VoH,fresnel):        
        FMi = ((-5.55473) * VoH - 6.98316) * VoH #interpolation
        return fresnel + (1.0-fresnel)*torch.pow(2.0,FMi)
    def cal_D(self,NoH,alpha):
        # specular D
        alpha2 = alpha*alpha.clamp_min_(1e-6)
        nom0 =(NoH*NoH*(alpha2-1)+ 1).clamp_min_(1e-3)# nom of D
        return alpha2/(nom0*nom0*np.pi) #(npt,nlights)
    def cal_G(self,NoL,NoV,roughness):
        k = (roughness*roughness+2*roughness+1/8.0)
        return 1/((NoV * (1 - k) + k) *(NoL * (1 - k) + k)).clamp_min_(1e-6)
    def cal_diffuse(self,albedo,fresnel):
        if len(fresnel.shape)>0 and(len(fresnel.shape)<len(albedo.shape)):
            fresnel = fresnel.unsqueeze(dim=-1).contiguous()
        return albedo/np.pi
    def forward(self,light_dir,view_dir,normal=None,albedo=None,roughness=None,fresnel=None):
        #all input should be flattened in spacial level 
        L = F.normalize(light_dir,dim=-1)#ray_num,light_num,3
        V = F.normalize(view_dir,dim=-1)

        albedo = self.albedo if albedo ==None else albedo
        alpha = self.roughness.repeat(1,1,3).contiguous() if roughness ==None else roughness
        fresnel = self.fresnel if fresnel == None else fresnel
        normal = F.normalize(self.normal,dim=-1) if normal == None else normal
        if self.hidx != None:
            albedo = albedo[self.hidx,...]
            fresnel = fresnel[self.hidx,...]
            alpha = alpha[self.hidx,...]
            normal = normal[self.hidx,...]
        else:
            albedo = albedo.view(-1,3).contiguous()
            fresnel = fresnel.view(-1,3).contiguous()
            alpha = alpha.view(-1,3).contiguous()
            normal = normal.view(-1,3).contiguous()


        
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
        specular_D = self.cal_D(dot_prod(N,H),alpha*alpha)
        specular_G = self.cal_G(dot_prod(N,L),dot_prod(N,V),alpha)
        specular_F = self.cal_F(dot_prod(V,H),fresnel)
        spec = (specular_D*specular_G*specular_F)/(4.0*dot_prod(N,V).clamp_min_(1e-6))
        if len(spec.shape)<len(albedo.shape):
            spec = spec.unsqueeze_(dim=-1).contiguous()
        brdf = self.cal_diffuse(albedo,fresnel)+spec
        return brdf.float()


class GGXRenderer_optim(GGXRenderer):
    def __init__(self,multilight=False,h=256,device='cuda') -> None:
        super(GGXRenderer_optim,self).__init__(multilight=multilight)
        self.albedo = torch.rand((h,h,3),dtype=torch.float32,device=device,requires_grad=True)
        self.roughness = torch.rand((h,h,1),dtype=torch.float32,device=device,requires_grad=True)
        self.fresnel = torch.rand((h,h,3),dtype=torch.float32,device=device,requires_grad=True)
        self.normal = torch.randn((h,h,3),dtype=torch.float32,device=device,requires_grad=True)
        self.height = h
        self.multilight = multilight
    def initialization(self,normal,albedo,roughness,fresnel):
        init_f = 0.5
        self.albedo.data[:] =  (albedo + torch.randn_like(albedo)*init_f).clamp(0.0,1.0)
        self.roughness.data[:]  = (roughness.unsqueeze(-1) + torch.randn_like(roughness.unsqueeze(-1))*init_f).clamp(0.0,1.0)
        self.normal.data[:]  = F.normalize(normal + torch.randn_like(normal)*init_f,dim=-1)
        self.fresnel.data[:]  = (fresnel + torch.randn_like(fresnel)*init_f).clamp(0.0,1.0)
    def clamp(self):
        self.albedo.data[:] =  (self.albedo).clamp(0.0,1.0)
        self.roughness.data[:]  = (self.roughness).clamp(0.0,1.0)
        self.normal.data[:]  = F.normalize(self.normal,dim=-1)
        self.fresnel.data[:]  = (self.fresnel).clamp(0.0,1.0)
    def rendering(self,light_dir,view_dir,light_rgb,normal=None,albedo=None,roughness=None,fresnel=None):
        '''img = torch.zeros_like(self.albedo).to(light_dir.device)
        for hidx in range(self.height):
            self.hidx = hidx
            img[hidx,...] = self.forward(light_dir,view_dir,normal,albedo,roughness,fresnel)'''
            
        img = self.forward(light_dir.view(-1,3),view_dir.view(-1,3),normal,albedo,roughness,fresnel)
        img =(light_rgb* np.pi*img).view(*(self.albedo.shape)).contiguous().clamp_(1e-10,1.0)
        prop = torch.cat([F.normalize(self.normal,dim=-1),self.albedo,self.roughness.repeat(1,1,3),self.fresnel],dim=-1)
        return img,prop

def get_rendering(wi,wo,data,Render,light_rgb,diffuse=True):
    if data.shape[-1] ==12:
        normal,albedo,alpha,fresnel = data.split([3,3,3,3],dim=-1)
    if data.shape[-1] ==10:
        normal,albedo,alpha,fresnel = data.split([3,3,1,3],dim=-1)
    if (data.shape[-1]==9):
        normal,albedo,fresnel = data.split([3,3,3],dim=-1)
        alpha = None
    if (data.shape[-1] ==7) and diffuse:
        normal,albedo,alpha= data.split([3,3,1],dim=-1)
        fresnel = None
    if (data.shape[-1] ==7) and (not diffuse):
        normal,alpha,fresnel= data.split([3,1,3],dim=-1)
        albedo = None
    if (data.shape[-1] == 6):
        normal,albedo = data.split([3,1,3],dim=-1)
        alpha = fresnel = None
    img = (light_rgb* np.pi*Render(wi,wo,normal,albedo,alpha,fresnel))
    img = img.view(*(albedo.shape)).contiguous().clamp_(1e-3,1.0) 
    return img



