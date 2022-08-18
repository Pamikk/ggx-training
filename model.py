from turtle import forward
import torch
from torch import nn

from render import GGXRenderer, get_rendering
def DX(x):
    return x[:,:,1:,:] - x[:,:,:-1,:]    # so this just subtracts the image from a single-pixel shifted version of itself (while cropping out two pixels because we don't know what's outside the image)
def DY(x):
    return x[:,1:,:,:] - x[:,:-1,:,:]    # likewise for y-direction

import torch.nn as nn
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
DROPOUT = 0.2
LReLU = 0.2
INPLACE = True
def conv3x3(in_channels, out_channels, stride=1,bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=bias)

#bias will be added in normalization layer
class BasicBlock(nn.Module):
    def __init__(self,in_channels,channels):
        super(BasicBlock,self).__init__()
        self.conv = conv3x3(in_channels,channels,bias=True)
        self.norm = nn.InstanceNorm2d(channels)
        self.relu = nn.LeakyReLU(LReLU,inplace=INPLACE)
    def forward(self,x):   
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)         
        return x
class BasicBlock_bn(nn.Module):
    def __init__(self,in_channels,channels):
        super(BasicBlock_bn,self).__init__()
        self.conv = conv3x3(in_channels,channels)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(LReLU,inplace=INPLACE)
    def forward(self,x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
class Encoder(nn.Module):
    def __init__(self,in_channel,channels,depth=1,dropout=False):
        super(Encoder,self).__init__()
        self.downsample = not(in_channel == channels)
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size = 2)
            self.conv = conv1x1(in_channel,channels,bias=True)
        seq=[nn.Sequential(BasicBlock(channels,channels),BasicBlock(channels,channels)) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
    def forward(self,x):
        if self.downsample:
            x = self.pool(x)
            x = self.conv(x)
        return self.seq(x)
class Encoder_bn(Encoder):
    def __init__(self,in_channel,channels,depth=1,dropout=False):
        super(Encoder_bn,self).__init__(in_channel,channels,depth,dropout)
        seq=[nn.Sequential(BasicBlock_bn(channels,channels),BasicBlock_bn(channels,channels)) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)

class Decoder(nn.Module):
    def __init__(self,channels,out_channel,depth=1):
        super(Decoder,self).__init__()
        self.upsample = not(out_channel == channels)
        if self.upsample:
            self.conv = conv1x1(channels,out_channel,bias=True)
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        seq=[nn.Sequential(BasicBlock(channels,channels),BasicBlock(channels,channels)) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
    def forward(self,x):
        x = self.seq(x)
        if self.upsample:
            x = self.conv(x)
            x = self.up(x)
        return x
class Decoder_bn(Decoder):
    def __init__(self,channels,out_channel,depth=1):
        super(Decoder_bn,self).__init__(channels,out_channel,depth)
        seq=[nn.Sequential(BasicBlock_bn(channels,channels),BasicBlock_bn(channels,channels)) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
Coders = {'base':(Encoder,Decoder),'bn':(Encoder_bn,Decoder_bn)}
def NetAPI(cfg,init=True):
    return Network(cfg)
class Network(nn.Module):
    def __init__(self,cfg):
        super(Network,self).__init__()
        channels = cfg.channels
        out_channels = []
        #softmax or sigmoid output
        encoder,decoder = Coders[cfg.net]
        self.in_channel = channels[0]
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        self.in_conv = conv1x1(3,self.in_channel)
        
        for channel in channels:
            out_channels.insert(0,self.in_channel)
            self.encoders.append(encoder(self.in_channel,channel,depth=cfg.depth))
            self.in_channel = channel
            
        for channel in out_channels:
            self.decoders.append(decoder(self.in_channel,channel))
            self.in_channel = channel
        if cfg.share:
                self.pred_normal = self.make_pred_layers(self.in_channel,3,'tanh')
                self.pred_svbrdf = self.make_pred_layers(self.in_channel,7,'sigmoid')
        else:
            self.normal_pred = self.make_pred_layers(self.in_channel,3,'tanh')
            self.albedo_pred = self.make_pred_layers(self.in_channel,3,'sigmoid')
            self.roughness_pred = self.make_pred_layers(self.in_channel,1,'sigmoid')
            self.fresnel_pred = self.make_pred_layers(self.in_channel,3,'sigmoid')
        self.share = cfg.share
        self.multilight = cfg.multilight
        self.usegt = cfg.usegt
    def make_pred_layers(self,in_channel,out_channel,activ):
        layers = []
        for i in [64,32]:
            layers+= [conv3x3(in_channel,i,bias=True), nn.LeakyReLU(LReLU,inplace=INPLACE)]
            in_channel=i
        if activ == 'tanh':
            layers+=[conv3x3(in_channel,out_channel,bias=True), nn.Tanh()]
        else:
            layers+=[conv3x3(in_channel,out_channel,bias=True), nn.Sigmoid()]
        return nn.Sequential(*layers)
    def forward(self,wi,wo,light_rgb,x,data_gt=None):
        feats = []
        x = x.permute([0,3,1,2])
        #x = x/2.0 -1.0
        x = self.in_conv(x)
        for i,encoder in enumerate(self.encoders):
            if i!=0:
                feats.insert(0,x)
            x = encoder(x) 
        for i,decoder in enumerate(self.decoders):
            x = decoder(x)
            if i<(len(feats)):
                x+=feats[i]
        if torch.isnan(x).any():
            print('nan in feat')
            exit()
        bs,_,h,w = x.shape
        if self.share:
            normal = self.pred_normal(x)
            svbrdf = self.pred_svbrdf(x)
            normal = F.normalize(normal,dim=1)
            data = torch.cat([normal,#torch.cat((data[:,:2,...],torch.ones((bs,1,h,w),dtype=feats.dtype,device=feats.device)),dim=1),dim=1).contiguous(),
                             svbrdf[:,:3,...].clamp(0.001,1.0),svbrdf[:,3,...].clamp(0.001,1.0).unsqueeze(1).repeat(1,3,1,1).contiguous(),svbrdf[:,4:,...].clamp(0.0,1.0)],dim=1)        
        else:
            normal =  F.normalize(self.normal_pred(x),dim=1)
            albedo = self.albedo_pred(x)
            roughness = self.roughness_pred(x).clamp(0.001,1.0).repeat(1,3,1,1).contiguous()
            fresnel =self.fresnel_pred(x)
            data = torch.cat((normal,albedo,roughness,fresnel),dim=1)
        data = torch.permute(data,[0,2,3,1]).contiguous()
        if torch.isnan(data).any():
            print('nan in data')
            exit()
        for i,usegt in enumerate(self.usegt):
            if usegt:
                data[...,3*i:3*(i+1)] = data_gt[...,3*i:3*(i+1)] 
        #print(data.shape)
        imgs = torch.zeros((bs,h,w,3),dtype=x.dtype,device=x.device)
        for i in range(bs):
            for j in range(h):
                if len(wi.shape) == len(data.shape):
                    imgs[i,j,...] = get_rendering(wi[i,j,...],wo[i,j,...],data[i,j,...],GGXRenderer(multilight=self.multilight),light_rgb[i,j,...])
                else:
                    imgs[i,j,...] = get_rendering(wi[i,...],wo[i,...],data[i,j,...],GGXRenderer(multilight=self.multilight),light_rgb[i,...])
        return imgs,data
MSE_Loss = torch.nn.MSELoss()
L1_Loss = torch.nn.L1Loss()
def log_mse(x,y):
    return MSE_Loss(torch.log(x+0.1),torch.log(y+0.1))
class RenderLoss(nn.Module):
    def __init__(self,type='l1',diff=True):
        super(RenderLoss,self).__init__()
        self.loss_type = type
        self.diff = diff
        self.sup_props = (True,True,True,True)
    def forward(self,img,gt_img,prop=None,gt_prop=None):
        assert img.shape == gt_img.shape
        if self.loss_type == 'l1':
            Loss_func = L1_Loss
        else:
            Loss_func = MSE_Loss
        loss_diff_x = L1_Loss(DX(img),DX(gt_img))
        loss_diff_y = L1_Loss(DY(img),DY(gt_img))

        render_loss = log_mse(img,gt_img)
        render_mse = MSE_Loss(img,gt_img) #for psnr calculation
        if self.diff:
            total_loss = render_loss+loss_diff_x+loss_diff_y
        else:
            total_loss = render_loss
        res = {'total':total_loss.item(),
                'diff_x':loss_diff_x.item(),
                'diff_y':loss_diff_y.item(),
                'render':render_loss.item(),
                'render_mse':render_mse.item()}
        propn = ['normal','albedo','roughness','fresnel']
        for i,flag in enumerate(self.sup_props):
            if flag:
                if i==0:
                    loss = F.huber_loss(prop[...,:3],gt_prop[...,:3])
                elif i==1:
                    loss = log_mse(prop[...,6],gt_prop[...,6])
                else:
                    loss = log_mse(prop[...,3*i:3*i+3],gt_prop[...,3*i:3*i+3])
                total_loss+= loss
                res[propn[i]] = loss.item()
        res['total']  = total_loss.item()       
        return total_loss,res
mse2psnr = lambda x: -10.0 * torch.log(x) / np.log(10.0)
def cal_rre(pred,gt):
    rre = torch.arccos(torch.clamp(torch.sum(gt * pred, dim=-1), -1.0, 1.0))
    rre = torch.mean(rre) / np.pi * 180
    return rre
class RenderMetric(nn.Module):
    def __init__(self):
        super(RenderMetric,self).__init__()
    def forward(self,img,gt_img,properties,gt_properties):
        metrics={}
        render_mse = MSE_Loss(img,gt_img)
        metrics['psnr'] = mse2psnr(render_mse).item()
        metrics['rmse'] = torch.sqrt(render_mse).item()
        #print(gt_properties.shape,properties.shape)
        normal_rre = cal_rre(properties[...,:3],gt_properties[...,:3])
        
        metrics['normal_rre'] = normal_rre.item()

        mse = MSE_Loss(properties[...,3:6],gt_properties[...,3:6])
        metrics['armse'] = torch.sqrt(mse).item()
        
        mse = MSE_Loss(properties[...,6:9],gt_properties[...,6:9])
        if torch.isnan(mse):
            print(properties.min(),gt_properties[...,6:9].min())
            exit()
        metrics['rrmse'] =  torch.sqrt(mse).item()

        mse = MSE_Loss(properties[...,9:],gt_properties[...,9:])
        metrics['frmse'] = torch.sqrt(mse).item()
        return metrics
        
        