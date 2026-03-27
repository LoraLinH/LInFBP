
import glob 
import re 
import pydicom
import h5py
import numpy as np


def findFiles(path): return glob.glob(path)

def image_read(image_path, image_type):
    if image_type in ['Mayo', 'Mayo_test']:
        image = pydicom.dcmread(image_path)
        imgdata = image.pixel_array * image.RescaleSlope + image.RescaleIntercept
        return imgdata
    elif image_type in ['MayoRaw', 'MayoRaw_test']:
        image = h5py.File(image_path, 'r')
        if 'img_x' in image:
            return np.transpose(image['img_x']) 
        elif  'sino_fan' in image:
            return np.transpose(image['sino_fan'])

def pop_paths(paths, num):
    for path in paths:
        for i in range(num):
            path.pop(0)
            path.pop()
    return paths


def findpath(path, idx, image_type):
    nums = re.findall('\d+', path)
    if image_type in ['Mayo', 'Mayo_test']:
        new_path = path.replace('_{}.IMA'.format(nums[-1]), '_{}.IMA'.format(int(nums[-1])+idx))
    elif image_type in ['MayoRaw', 'MayoRaw_test']:
        new_path = path.replace('_{}.mat'.format(nums[-1]), '_{}.mat'.format(int(nums[-1])+idx))
    return new_path


def calc_psnr(img_ref, img_eva):
    mse = np.mean((img_ref.astype(float) - img_eva.astype(float)) ** 2)
    psnr = 10 * np.log10(np.max(img_eva.astype(float))**2 / mse)
    # psnr = 10 * np.log10(255*^^*2 / mse)
    return psnr

# def calc_psnr(img_ref, img_eva):
#     mse = np.std(img_ref.astype(float) - img_eva.astype(float))
#     psnr = 20 * np.log10(np.max(img_eva.astype(float)) / mse)
#     # psnr = 10 * np.log10(255**2 / mse)
#     return psnr


# def calc_psnr(img_ref, img_eva):
#     numer = np.sqrt(np.mean((img_ref.astype(float) - img_eva.astype(float)) ** 2))
#     denom = np.sqrt(np.mean(img_eva.astype(float) ** 2))
#
#     nmse = numer / denom
#     return nmse

def calc_nmse(img_ref, img_eva):

    # Calc NMSE
    # Grayscale ver.
    numer = np.sqrt(np.sum((img_ref.astype(float) - img_eva.astype(float)) ** 2))
    denom = np.sqrt(np.sum(img_eva.astype(float) ** 2))

    nmse = numer / denom
    return nmse


