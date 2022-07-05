import torch
import math
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
mse2psnr = lambda x: -10.0 * np.log(x) / np.log(10.0)
from torch.nn import functional as F
def cal_rre(pred,gt):
    rre = np.arccos(np.clip(np.sum(gt * pred, axis=-1), -1.0, 1.0))
    rre = np.mean(rre) / np.pi * 180
    return rre
class Logger(object):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.files = {'val':open(os.path.join(log_dir,'val.txt'),'a+'),'train':open(os.path.join(log_dir,'train.txt'),'a+')}
        self.eventwriter = SummaryWriter(log_dir=self.log_dir)
    def write_line2file(self,mode,string):
        self.files[mode].write(string+'\n')
        self.files[mode].flush()
    def write_loss(self,epoch,losses,lr):
        tmp = str(epoch)+'\t'+str(lr)+'\t'
        print('Epoch',':',epoch,'-',lr)
        writer = self.eventwriter
        writer.add_scalar('lr',math.log(lr),epoch)
        for k in losses:
            if losses[k]>0:            
                writer.add_scalar('Train/'+k,losses[k],epoch)            
                print(k,':',losses[k])
                #self.writer.flush()
        tmp+= str(round(losses['total'],5))+'\t'
        self.write_line2file('train',tmp)
        writer.flush()
    def write_runningloss(self,step,losses,lr):
        writer = self.eventwriter
        writer.add_scalar('running lr',math.log(lr),step) 
        for k in losses:
            writer.add_scalar('Running/'+k,losses[k],step)  
        writer.flush()
    def write_metrics(self,epoch,metrics,save=[],mode='Val',log=True):
        tmp =str(epoch)+'\t'
        print("validation epoch:",epoch)
        writer = self.eventwriter 
        for k in metrics:
            if k in save:
                tmp +=str(metrics[k])+'\t'
            if log:
                tag = mode+'/'+k            
                writer.add_scalar(tag,metrics[k],epoch)
                #self.writer.flush()
            print(k,':',metrics[k])
        
        self.write_line2file('val',tmp)
        writer.flush()
def generate_normalized_random_direction(batchSize, lowEps = 0.001, highEps =0.05,device = 'cuda'):
    r1 = torch.random_uniform([batchSize, 1], 0.0 + lowEps, 1.0 - highEps, dtype=torch.float32)
    r2 =  torch.random_uniform([batchSize, 1], 0.0, 1.0, dtype=torch.float32)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2
       
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - torch.square(r))
    finalVec = torch.concat([x, y, z], axis=-1) #Dimension here is [batchSize, 3]
    return finalVec.cuda() if device=='cuda' else finalVec 
def resize_stack_images(pred):
    assert pred.shape[-1] == 15
    imgs = list(pred.split([3,3,3,3,3],dim=-1))
    imgs[1] = (imgs[1]+1.0)/2.0
    return torch.cat(imgs,dim=1)
def vis_batch(pred,gt):
    pred = pred.detach().cpu().squeeze()
    gt = gt.detach().squeeze()

    pred_img = resize_stack_images(pred)
    gt_img = resize_stack_images(gt)
    img = (torch.cat((pred_img,gt_img),dim=0).clamp(0.0,1.0)).numpy()*255
    return img


