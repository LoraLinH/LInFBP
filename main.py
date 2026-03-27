import os
import numpy as np 
np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=np.nan)
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import pickle
# from torch.utils.data.dataloader import default_collate
from scipy.signal.windows import hann, cosine
import torch 
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from Utils.initParameter import InitPara
from Utils.initFunction import weights_init
from Datasets.imageProcess import Transpose, TensorFlip, MayoTrans, SinoTrans
from Datasets.datasets import TrainDataSet
# from Solver.pixelIndexCal_cuda import PixelIndexCal_cuda
from Solver.train import train_model
from Solver.test import test_model
from Model.model_fbp_L import FBP as FBPModel



def main():

    opt = InitPara()

    geo = {'nVoxelX': 512, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
                'nVoxelY': 512, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
                'nDetecU': 736, 'sDetecU': 0.6848*2*736, 'dDetecU': 0.6848*2,
                'offOriginX': 0.0, 'offOriginY': 0.0,
                'views': 100, 'slices': 1,
                'DSD': 1085.6, 'DSO': 595.0, 'DOD': 490.6,
                'start_angle': 0.0, 'end_angle': 2*np.pi,
                'mode': 'fanflat', 'extent': 1, # currently extent supports 1, 2, or 3.
                }



    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    post_trans_img = MayoTrans(opt.WaterAtValue, trans_style='image')
    post_trans_sino = SinoTrans(trans_style='sino')

    print('Constructing Datasets...')
    if opt.is_train:
        datasets = {'train': TrainDataSet(opt.root_path, opt.TrainFolder, geo, None if opt.Dataset_name == 'MayoRaw' else pre_trans_img, post_trans_img, post_trans_sino, opt.Dataset_name),
                    'val': TrainDataSet(opt.root_path, opt.ValFolder, geo, None if opt.Dataset_name == 'MayoRaw' else pre_trans_img, post_trans_img, post_trans_sino, opt.Dataset_name)}
        
        dataloaders = {x: DataLoader(datasets[x], opt.batch_size[x], shuffle=True if x =='train' else False, pin_memory=True, num_workers=opt.num_workers[x]) for x in ['train', 'val']}
        dataset_sizes = {x: opt.batch_num[x]*opt.batch_size[x] for x in ['train', 'val']}
    else:
        datasets = {'test': TrainDataSet(opt.root_path, opt.TestFolder, geo, None, post_trans_img, post_trans_sino, 'Mayo_test' if opt.Dataset_name == 'Mayo' else 'MayoRaw_test')}
        dataloaders = {x: DataLoader(datasets[x], opt.batch_size[x], shuffle=opt.is_shuffle, pin_memory=True, num_workers=opt.num_workers[x]) for x in ['test']}
        dataset_sizes = {x: opt.batch_num[x]*opt.batch_size[x] for x in ['test']}
    print('Done!')

    w = (geo['nDetecU'] - 1) / 2
    s = geo['dDetecU'] * (np.arange(geo['nDetecU']) - w)
    gam = np.arctan(s / geo['DSD'])
    w1 = np.abs(geo['DSO'] * np.cos(gam) - 0 * np.sin(gam)) / geo['DSD']
    geo['w1'] = torch.from_numpy(w1).cuda()

    npad = 2 ** np.ceil(np.log2(2 * geo['nDetecU'] - 1))  # padded size
    npad = int(npad)
    nnp = np.arange(-(npad // 2), npad // 2)
    h = np.zeros_like(nnp, dtype=float)
    h[npad // 2] = 1 / 4
    odd = nnp % 2 == 1
    h[odd] = -1 / (np.pi * nnp[odd]) ** 2
    h /= geo['dDetecU'] ** 2
    Hk = np.real(fft(fftshift(h)))

    window = np.ones((npad))
    # window = hann(npad)
    window = fftshift(window)
    Hk = Hk * window
    geo['filter'] = torch.from_numpy(Hk * geo['dDetecU']).cuda()

    betas = np.linspace(geo['start_angle'], geo['end_angle'], geo['views'], False)
    betas = np.expand_dims(np.expand_dims(betas, 0), 0)
    xc = np.arange(1, geo['nVoxelX'] + 1) - (geo['nVoxelX'] + 1) / 2
    yc = np.arange(1, geo['nVoxelY'] + 1) - (geo['nVoxelY'] + 1) / 2
    yc = np.flip(yc)
    xc = np.expand_dims(np.expand_dims(xc, -1), 0) * geo['dVoxelX']
    yc = np.expand_dims(np.expand_dims(yc, -1), -1) * geo['dVoxelY']
    d_loop = geo['DSO'] - xc * np.sin(betas) + yc * np.cos(betas)  # dso - y_beta
    mag = geo['DSD'] / d_loop
    geo['w2'] = torch.from_numpy(mag ** 2).cuda()  # [np] image-domain weighting

    indices_path = 'Results/test_512_{}_fan.dat'.format(geo['views'])
    if os.path.isfile(indices_path):
        print('Loading sinoIndices...')
        geo['indices'] = pickle.load(open(indices_path, "rb"))
        print('Done!')
    else:
        print('Generating sinoIndices...')
        geo['indices'], _ = PixelIndexCal_cuda(geo_virtual)
        f = open(indices_path, "wb")
        pickle.dump(geo['indices'], f, True)
        f.close()
        print('Done!')

    if opt.use_cuda:
        geo['indices'] = geo['indices'].cuda()
        print('CUDA!')

    net = FBPModel(geo, opt)
    criterion = nn.MSELoss()

    if not opt.reload_model:
        min_loss = None
        pre_losses = None
    else:
        # epoch_reload_path = opt.target_folder + 'Model_save/{}.pkl'.format(model_name)
        epoch_reload_path = "Results/Model_save/best_val_model_{}.pkl".format(opt.net_name)
        if os.path.isfile(epoch_reload_path):
            print('Loading previously trained network: {}...'.format(epoch_reload_path))
            checkpoint = torch.load(epoch_reload_path, map_location = lambda storage, loc: storage)
            # model_dict = net.state_dict()
            # checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
            # model_dict.update(checkpoint)
            model_dict = checkpoint
            net.load_state_dict(model_dict)
            del checkpoint
            torch.cuda.empty_cache()
            print('Done!')

        min_loss_path = opt.target_folder + "Loss_save/min_loss_{}.dat".format(opt.net_name) if opt.is_train else ''
        min_loss = pickle.load(open(min_loss_path, "rb")) if os.path.isfile(min_loss_path) else None
        pre_losses_path = opt.target_folder + "Loss_save/losses_{}.dat".format(opt.net_name) if opt.is_train else ''
        pre_losses = pickle.load(open(pre_losses_path, "rb")) if os.path.isfile(pre_losses_path) else None

    if opt.use_cuda:
        net.cuda()
        criterion.cuda()

    optimizer_ft = optim.RMSprop(net.parameters(), lr=2e-5, momentum=0.9, weight_decay=0.0)

    if opt.is_train and opt.reload_model:
        optimizer_name = "backup_optimizer_{}".format(opt.net_name)
        optimizer_reload_path = opt.target_folder + "Optimizer_save/{}.pkl".format(optimizer_name)
        if os.path.isfile(optimizer_reload_path):
            print('Loading previous optimizer: {}...'.format(optimizer_name))
            checkpoint = torch.load(optimizer_reload_path, map_location = lambda storage, loc: storage)
            optimizer_ft.load_state_dict(checkpoint)
            del checkpoint
            torch.cuda.empty_cache()
            print('Done!')

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=0.5) if opt.is_lr_scheduler else None


    if opt.is_train:
        train_model(dataloaders, net, optimizer_ft, criterion, exp_lr_scheduler, min_loss, pre_losses, num_epochs=15, dataset_sizes=dataset_sizes, opt=opt)
    else:
        net.eval()
        test_model(dataloaders, net, criterion, opt)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main()

