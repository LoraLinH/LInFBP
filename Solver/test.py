

import pickle
import torch
import time 
from torch.autograd import Variable 
import scipy.io
from Datasets.imageProcess import DeMayoTrans, DeSinoTrans
import re
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from Datasets.utils import calc_nmse, calc_psnr
from Datasets.cal_fsim import FeatureSIM
from tqdm import tqdm
import cv2
# from skimage.metrics import peak_signal_noise_ratio as psnr


def test_model(dataloaders, model, criterion=None, opt=None):

    nmse = []
    psnr = []
    fsim2 = []

    post_recon_trans_img = DeMayoTrans(opt.WaterAtValue, trans_style='image')
    # post_recon_trans_sino = DeSinoTrans(trans_style='sino')
    for i_batch, data in enumerate(tqdm(dataloaders['test'])):
        if i_batch == opt.batch_num['test']:
            break
        inputs, labels = data['sinogram'], data['ndct']

        # torch.cuda.reset_peak_memory_stats()

        if opt.use_cuda:  # wrap them in Variable
            labels = Variable(labels).cuda()
            inputs = Variable(inputs).cuda()
        else:
            labels = Variable(labels)
            inputs = Variable(inputs)

        with torch.no_grad():
            outputs = model(inputs)

        final_output = (outputs[0][0].detach().cpu().numpy())
        final_labels = (labels[0][0].detach().cpu().numpy())

        final_output2 = np.clip((final_output - np.min(final_labels)) / np.ptp(final_labels), 0, 1) * 255
        final_labels2 = (final_labels - np.min(final_labels)) / np.ptp(final_labels) * 255

        nmse.append(calc_nmse(final_output, final_labels))
        psnr.append(calc_psnr(final_output, final_labels))
        fsim2.append(FeatureSIM(final_labels2, final_output2)[0])


    '''
    save_dir = r'E:\new01\TMI_kernel\ModelResults\{}.txt'.format(opt.net_name)
    with open(save_dir, 'w') as f:
        for p in psnr:
            f.write(str(p))
            f.write('\n')
        f.write('---------------\n')
        for n in nmse:
            f.write(str(n))
            f.write('\n')
        f.write('---------------\n')
        for fs in fsim2:
            f.write(str(fs))
            f.write('\n')
    '''

    psnr_rlt_avg = sum(psnr) / len(psnr)
    nmse_rlt_avg = sum(nmse) / len(nmse)
    fsim2_rlt_avg = sum(fsim2) / len(fsim2)

    psnr_rlt_std = np.std(np.array(psnr))
    nmse_rlt_std = np.std(np.array(nmse))
    fsim2_rlt_std = np.std(np.array(fsim2))

    print('# Validation # PSNR: {:.4e}:'.format(psnr_rlt_avg))
    print('# Validation # NMSE: {:.4e}:'.format(nmse_rlt_avg))
    print('# Validation # FSIM2: {:.4e}:'.format(fsim2_rlt_avg))

    print('# Validation # PSNR_STD: {:.4e}:'.format(psnr_rlt_std))
    print('# Validation # NMSE_STD: {:.4e}:'.format(nmse_rlt_std))
    print('# Validation # FSIM2_STD: {:.4e}:'.format(fsim2_rlt_std))
