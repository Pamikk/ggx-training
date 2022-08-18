from unittest import TestLoader
import torch
from dataset import Material
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import yaml
from render import GGXRenderer_optim
from optimizer import Trainer
class Config:
    def __init__(self):
        self.mode = "train" #only test/train in this exp
        exp_path = "/home/pami/exps/ggx_scratch"
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        self.exp_name = "optimize_rand_log" # diff
        self.log_exp_path = os.path.join(exp_path,self.exp_name)
        if not os.path.exists(self.log_exp_path):
            os.mkdir(self.log_exp_path)
        
        #data
        self.data_path = "/home/pami/dataset/materialsData_multi_image"
        self.multilight = False
        # train
        self.deterministic = True
        self.resume =0
        self.epochs = 2000
        self.ovf = True
        self.bs = 1 if self.ovf else 8

        self.val_every_k_epoch = self.epochs//100
        self.save_every_k_epoch = self.epochs//5
        self.lr = 0.005
        self.weight_decay = 0.0
        self.lr_factor = 0.1
        self.schedules = np.array([4000,8000,20000])//self.bs
        self.patience = 20
        print(self.patience)
        self.min_lr = 1e-7
        #network
        self.net = 'base' #base
        self.channels = [64,128,256,512]
        self.share = True
        self.depth = 1
        self.device = 'cuda'
        self.loss_type = 'mse'
        self.diff = False

        

def main():
    np.random.seed(2333)
    
    cfg = Config()
    torch.manual_seed(2333)
    torch.set_printoptions(precision=10)
    model = GGXRenderer_optim()

    TrainLoader = DataLoader(Material(path=cfg.data_path,mode='train',multilight=cfg.multilight,overfitting=cfg.ovf),batch_size=cfg.bs,shuffle=True,num_workers=8)
    
    TrainvalLoader = DataLoader(Material(path=cfg.data_path,mode='trainval',multilight=cfg.multilight,overfitting=cfg.ovf),batch_size=1,shuffle=True)
    if cfg.ovf:
        TestLoader = DataLoader(Material(path=cfg.data_path,mode='test',multilight=cfg.multilight,overfitting=cfg.ovf),batch_size=1,shuffle=False)
        datasets ={'train':TrainLoader,'test':TestLoader,'trainval':TrainvalLoader}
    else:
        ValLoader= TestLoader = DataLoader(Material(path=cfg.data_path,mode='test',multilight=cfg.multilight),batch_size=1,shuffle=False)
        datasets ={'train':TrainLoader,'test':TestLoader,'trainval':TrainvalLoader,'val':ValLoader}
    trainer = Trainer(cfg,datasets,model,(cfg.resume,cfg.epochs))
    trainer.train()


    yaml.dump(cfg,open(os.path.join(cfg.log_exp_path,"config.yaml"),'w'))

if __name__ == "__main__":
    main()