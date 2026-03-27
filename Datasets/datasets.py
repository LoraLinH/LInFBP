
import astra
import torch 
import copy 
import numpy as np 
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom
import math
import pickle

from Datasets.utils import findFiles, pop_paths, findpath, image_read
from Datasets.imageProcess import CTnum2AtValue
from Datasets.imageProcess import Transpose
from scipy import io
import matplotlib.pyplot as plt


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        return torch.from_numpy(image).type(torch.FloatTensor)


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class TensorFlip(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, image):
        return flip(image, self.dim)


class TrainData(Dataset):
    def __init__(self, root_dir, folder, crop_size=None, recon_slices=None, trf_op=None, Dataset_name='Mayo_test'):
        self.Dataset_name = Dataset_name
        self.trf_op = trf_op
        self.crop_size = crop_size
        self.recon_slices = recon_slices
        self.WaterAtValue = 0.0192
        self.hd_image_paths = [findFiles(root_dir+'/{}/{}_{}/*.IMA'.format(x, y, z)) for x in folder['patients'] for y in folder['HighDose'] for z in folder['SliceThickness']]
        self.hd_image_paths = sorted([x for j in self.hd_image_paths for x in j])
        self.ld_image_paths = [findFiles((root_dir+'/{}/{}_{}/*.IMA'.format(x, y, z)).replace('full', 'quarter')) for x in folder['patients'] for y in folder['LowDose'] for z in folder['SliceThickness']]
        self.ld_image_paths = sorted([x for j in self.ld_image_paths for x in j])

    
    def __len__(self):

        return len(self.hd_image_paths)

    
    def __getitem__(self, idx):
        hd_image_path_mid = self.hd_image_paths[idx]
        ld_image_path_mid = self.ld_image_paths[idx]
        image = pydicom.dcmread(hd_image_path_mid)
        imgdata = image.pixel_array * image.RescaleSlope + image.RescaleIntercept
        # image.RescaleSlope 1; image.RescaleIntercept -1024


        ldimage = pydicom.dcmread(ld_image_path_mid)
        ldimgdata = ldimage.pixel_array * ldimage.RescaleSlope + ldimage.RescaleIntercept

        random_list = [ToTensor(), CTnum2AtValue(self.WaterAtValue)]

        transform = transforms.Compose(random_list)
        imgdata = transform(imgdata)
        ldimgdata = transform(ldimgdata)

        return imgdata, ldimgdata

class TrainDataSet(Dataset):
    def __init__(self, root_dir, folder, geo, pre_trans_img=None, post_trans_img=None, post_trans_sino=None, Dataset_name='Mayo_test'):
        self.Dataset_name = Dataset_name
        self.imgset = TrainData(root_dir, folder, geo['nVoxelX'], geo['slices'], pre_trans_img, Dataset_name)
        self.vol_geom = astra.create_vol_geom(geo['nVoxelY'], geo['nVoxelX'],
                                               -1*geo['sVoxelY']/2 + geo['offOriginY'], geo['sVoxelY']/2 + geo['offOriginY'], -1*geo['sVoxelX']/2 + geo['offOriginX'], geo['sVoxelX']/2 + geo['offOriginX'])
        self.proj_geom = astra.create_proj_geom(geo['mode'], geo['dDetecU'], geo['nDetecU'], 
                                                np.linspace(geo['start_angle'], geo['end_angle'], geo['views'],False), geo['DSO'], geo['DOD'])
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        # self.post_trans_img = post_trans_img
        # self.post_trans_sino = post_trans_sino
        self.post_trans_img = None
        self.post_trans_sino = None
        
    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):

        img, ldimg = self.imgset[idx]
        img = img.numpy()
        ldimg = ldimg.numpy()

        try:
            sinogram_id, sinogram = astra.create_sino(img, self.proj_id)
        except:
            self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
            sinogram_id, sinogram = astra.create_sino(img, self.proj_id)
        astra.data2d.delete(sinogram_id)

        # sinogram = ((sinogram - 6.9)/(-0.1 - 6.9))

        img, sinogram = np.array(img), np.array(sinogram)

        sample = {'ndct': torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0), 'sinogram': torch.from_numpy(sinogram).type(torch.FloatTensor).unsqueeze(0)}
        # sample = {'ndct': img.unsqueeze(0), 'sinogram': sinogram.unsqueeze(0)}

        return sample


        # if self.post_trans_img is not None:
        #     img, _, _ = self.post_trans_img(img)
        #
        # if self.post_trans_sino is not None:
        #     sinogram, _, _ = self.post_trans_sino(sinogram)
        # sinogram = sinogram / 2 - 0.5

        