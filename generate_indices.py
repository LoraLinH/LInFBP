import os
import numpy as np 
np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=np.nan)
import pickle
import torch


from Utils.initParameter import InitPara
from Solver.pixelIndexCal_cuda import PixelIndexCal_cuda



def main():

    opt = InitPara()

    geo_real = {'nVoxelX': 512, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
                'nVoxelY': 512, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
                'nDetecU': 736, 'sDetecU': 504.0128 * 2, 'dDetecU': 0.6848 * 2,
                'offOriginX': 0.0, 'offOriginY': 0.0,
                'views': 290, 'slices': 1,
                'DSD': 1085.6, 'DSO': 595.0, 'DOD': 490.6,
                'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
                'mode': opt.geo_mode, 'extent': 1,  # currently extent supports 1, 2, or 3.
                }

    print('Generating sinoIndices...')
    geo_real['indices'], weight = PixelIndexCal_cuda(geo_real)
    print(geo_real['indices'].size())

    f = open("Results/test_512_{}_fan.dat".format(geo_real['views']), "wb")
    pickle.dump(geo_real['indices'], f, True)
    f.close()
    print('Done!')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main()

