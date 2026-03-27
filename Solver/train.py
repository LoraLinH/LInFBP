

import pickle
import torch
import time 
import os
from torch.autograd import Variable
from Datasets.imageProcess import DeMayoTrans
import scipy.io
from Datasets.utils import calc_nmse, calc_psnr
import numpy as np
from tqdm import tqdm
from Datasets.cal_fsim import FeatureSIM
import cv2

def tvloss(x, y):
    h_tv = x[...,1:,:]-x[...,:-1,:]
    h_tv_y = y[..., 1:, :] - y[..., :-1, :]
    w_tv = x[...,1:] - x[...,:-1]
    w_tv_y = y[..., 1:] - y[..., :-1]
    tv_loss = torch.mean(torch.abs(h_tv-h_tv_y)) + torch.mean(torch.abs(w_tv-w_tv_y))
    return tv_loss

def train_model(dataloaders, model, optimizer, criterion=None, scheduler=None, min_loss=None, pre_losses=None, num_epochs=25, dataset_sizes=None, opt=None):
    since = time.time()
    best_psnr = 0
    post_recon_trans_img = DeMayoTrans(opt.WaterAtValue, trans_style='image')
        
    pre_epoch = 0
    epoch_loss = {x: 0.0 for x in ['train']}

    for epoch in range(pre_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        if scheduler is not None:
            scheduler.step()
        model.train()  # Set model to training mode

        running_loss = 0.0
        tmp_losses = torch.zeros(1, 0)
        for i_batch, data in enumerate(tqdm(dataloaders['train'])):
            if i_batch == opt.batch_num['train']:
                break

            labels = data['ndct']
            inputs = data['sinogram']
            if opt.use_cuda:  # wrap them in Variable
                labels = Variable(labels).cuda()
                inputs = Variable(inputs).cuda()
            else:
                labels = Variable(labels)
                inputs = Variable(inputs)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            tmp_losses = torch.cat((tmp_losses, torch.tensor([[loss.data.item()]])), 1)
            running_loss += loss.data.item() * inputs.size(0)


        epoch_loss['train'] = running_loss / (dataset_sizes['train'])
        print('{} Loss: {:.8f}'.format('train', epoch_loss['train']))

    torch.save(model.state_dict(), opt.target_folder + 'Model_save/best_{}_model_{}.pkl'.format('val', opt.net_name))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimun train loss: {:5f}'.format(min_loss['train']))

    