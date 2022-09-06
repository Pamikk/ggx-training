import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import random
from model import RenderLoss,RenderMetric
from utils import Logger, vis_batch
import cv2
class Trainer:
    def __init__(self,cfg,datasets,net,epoch):
        self.cfg = cfg
        self.valset = None
        if 'train' in datasets:
            self.trainset = datasets['train']
        if 'val' in datasets:
            self.valset = datasets['val']
        if 'trainval' in datasets:
            self.trainval = datasets['trainval']
        else:
            self.trainval = False
        if 'test' in datasets:
            self.testset = datasets['test']
        self.net = net
        self.loss = RenderLoss(cfg.loss_type,cfg.diff)
        self.eval_metric = RenderMetric()

        name = cfg.exp_name
        self.name = name
        self.checkpoints = cfg.log_exp_path

        self.device = cfg.device

        self.optimizer = optim.Adam(self.net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        #self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=cfg.lr_factor, threshold=1e-2,patience=cfg.patience,min_lr=cfg.min_lr)
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 7500*2//cfg.bs, T_mult=1, eta_min=5e-6, last_epoch=- 1, verbose=False)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,cfg.schedules,gamma=cfg.lr_factor)
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        self.predictions = os.path.join(self.checkpoints,'pred')
        if not(os.path.exists(self.predictions)):
            os.mkdir(self.predictions)

        start,total = epoch
        self.start = start        
        self.total = total
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate
        self.val_every_k_epoch = cfg.val_every_k_epoch
        self.upadte_grad_every_k_batch = 1

        self.best_metric = 0
        self.best_metric_epoch = 0
        self.movingLoss = 0
        self.bestMovingLoss = 1e9
        self.bestMovingLossEpoch = 1e9

        self.early_stop_epochs = 50
        self.alpha = 0.95 #for update moving loss
        self.lr_change= False


        self.save_pred = False
        self.steps = 0
        
        #load from epoch if required
        if start:
            if (start=='-1')or(start==-1):
                self.load_last_epoch()
            else:
                self.load_epoch(start)
        else:
            self.start = 0
        self.net = self.net.to(self.device)

    def load_last_epoch(self):
        files = os.listdir(self.checkpoints)
        idx = 0
        for name in files:
            if name[-3:]=='.pt':
                epoch = name[6:-3]
                if epoch=='best' or epoch=='bestm':
                  continue
                idx = max(idx,int(epoch))
        if idx==0:
            exit()
        else:
            self.load_epoch(str(idx))
    def save_epoch(self,idx,epoch):
        saveDict = {'net':self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':self.lr_scheduler.state_dict(),
                    'epoch':epoch,
                    'best_metric':self.best_metric,
                    'best_metric_epoch':self.best_metric_epoch,
                    'movingLoss':self.movingLoss,
                    'bestmovingLoss':self.bestMovingLoss,
                    'bestmovingLossEpoch':self.bestMovingLossEpoch,'steps':self.steps}
        path = os.path.join(self.checkpoints,f'{idx}.pt')
        torch.save(saveDict,path)                  
    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,f'epoch_{idx}.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            
        else:
            print(f'no such model at:{model_path}')
            model_path = os.path.join(self.checkpoints,f'{idx}.pt')
            if os.path.exists(model_path):
                print('load:'+model_path)
            else:
                print(f'And no such model at:{model_path}')
                exit()
        info = torch.load(model_path)
        self.net.load_state_dict(info['net'])
        if not(self.lr_change):
            self.optimizer.load_state_dict(info['optimizer'])#might have bugs about device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.lr_scheduler.load_state_dict(info['lr_scheduler'])
        self.start = info['epoch']+1
        self.best_metric = info['best_metric']
        self.best_metric_epoch = info['best_metric_epoch']
        self.movingLoss = info['movingLoss']
        self.bestMovingLoss = info['bestmovingLoss']
        self.bestMovingLossEpoch = info['bestmovingLossEpoch']
        self.steps = info['steps']
    def _updateRunningLoss(self,loss,epoch):
        if self.bestMovingLoss>loss:
            self.bestMovingLoss = loss
            self.bestMovingLossEpoch = epoch
            self.save_epoch('bestm',epoch)
            print(f'loss decrease: {self.bestMovingLoss}')
    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))
    def set_lr(self,lr):
        #adjust learning rate manually
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr
        #tbi:might set different lr to different kind of parameters
    def adjust_lr(self,lr_factor):
        #adjust learning rate manually
        for param_group in self.optimizer.param_groups:
            param_group['lr']*=lr_factor
    def check_grad_norm(self):
        total_norm=0
        max_norm=0
        
        for p in self.net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_norm = max(param_norm.item(),max_norm)
        total_norm = total_norm ** (1. / 2)
        print(total_norm,max_norm)
    def train_one_epoch(self):
        self.optimizer.zero_grad()
        running_loss ={'total':0.0,
                'diff_x':0.0,
                'diff_y':0.0,
                'render':0.0,
                'render_mse':0.0}
        running = 0.0
        self.net.train()
        n = len(self.trainset)
        torch.cuda.empty_cache()
        times ={'total':0.0,'data':0.0}
        start = time.time()
        for data in tqdm(self.trainset):
            times['data'] += (time.time()-start)/n
            wi,wo,light_rgb,imgs,props_gt = data
            preds,props = self.net(wi.to(self.device),wo.to(self.device),light_rgb.to(self.device),imgs.to(self.device),props_gt.to(self.device))
            loss,losses = self.loss(preds,imgs.to(self.device),props,props_gt.to(self.device))            
            for k in losses.keys():
                if np.isnan(losses[k]):
                        continue
                if k in running_loss:
                    running_loss[k] += losses[k]/n
                else:
                    running_loss[k] = losses[k]/n
            if torch.isnan(loss):
                print('nan loss')
                del imgs,preds,losses
                continue
            loss.backward()
            running = loss.item()
            self.steps+=1
            
            losses['watch'] = running
            self.logger.write_runningloss(self.steps,losses,self.optimizer.param_groups[0]['lr'])
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            del imgs,preds,losses
            #solve gradient explosion problem caused by large learning rate or small batch size
            #nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2.0)             
            #nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
            
            del loss
            times['total'] += (time.time()-start)/n
            start = time.time()
        print(f'run time:{times["total"]},data time:{times["data"]}')
        self.logMemoryUsage()
        return running_loss
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")
        self.optimizer.zero_grad()
        print(self.optimizer.param_groups[0]['lr'])
        epoch = self.start
        stop_epochs = 0
        while epoch < self.total and stop_epochs<self.early_stop_epochs:
            running_loss = self.train_one_epoch()          
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.write_loss(epoch,running_loss,lr)
            #step lr
            self._updateRunningLoss(running_loss['total'],epoch)
            lr_ = self.optimizer.param_groups[0]['lr']
            if lr_ == self.cfg.min_lr:
                stop_epochs +=1
            if (epoch+1)%self.save_every_k_epoch==0:
                self.save_epoch(f'epoch_{epoch+1}',epoch)
            if (epoch+1)%self.val_every_k_epoch==0:
                
                
                if self.trainval:
                    #eval train subset
                    metrics = self.validate(epoch,'train',self.save_pred)
                    tosave = list(metrics.keys())
                    self.logger.write_metrics(epoch,metrics,tosave,mode='Trainval')
                    psnr = metrics['psnr']
                if self.valset!=None:
                    metrics = self.validate(epoch,'val',self.save_pred)
                    tosave = list(metrics.keys())
                    self.logger.write_metrics(epoch,metrics,tosave)
                    psnr = metrics['psnr']
                if psnr >= self.best_metric:
                    self.best_metric = psnr
                    self.best_metric_epoch = epoch
                    print("best so far, saving......")
                    self.save_epoch('best',epoch)
            print(f"best so far with {self.best_metric} at epoch:{self.best_metric_epoch}")
            epoch +=1
                
        print("Best: {:.4f} at epoch {}".format(self.best_metric, self.best_metric_epoch))
        self.save_epoch(f'epoch_{epoch}',epoch)
    def validate(self,epoch,mode,save=False):
        print('start Validation Epoch:',epoch,mode)
        valset = self.valset if mode=='val' else self.trainval
        metrics = {}
        n = len(valset)
        idx = 0
        self.net.eval()
        for data in tqdm(valset):
            with torch.no_grad():
                wi,wo,light_rgb,imgs,prop_gt = data
                preds,props = self.net(wi.to(self.device),wo.to(self.device),light_rgb.to(self.device),imgs.to(self.device),prop_gt.to(self.device))
                result = self.eval_metric(preds,imgs.to(self.device),props,prop_gt.to(self.device))           
                for k in result:
                    if k in metrics.keys():
                        metrics[k].append(result[k])
                    else:
                        metrics[k]=[result[k]]
                img = vis_batch(torch.cat([preds,props],dim=-1),torch.cat([imgs,prop_gt],dim=-1))
                save_path = os.path.join(self.predictions,mode)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                
                idx+=1
            if idx==2:
                cv2.imwrite(os.path.join(save_path,f'vis_{idx}_{epoch}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_path,f'vis_{idx-1}_{epoch}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        self.logMemoryUsage()
        for k in metrics:
            metrics[k] = np.mean(metrics[k])
        return metrics
    def test(self):
        self.net.eval()
        res = []
        with torch.no_grad():
            pass
        self.logMemoryUsage()
        return res
    def validate_random(self):
        pass