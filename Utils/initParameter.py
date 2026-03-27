
import torch 
import os
import numpy as np 


class InitPara(object):
    def __init__(self):
        self.Dataset_name = 'Mayo' # type of training dataset
        self.use_cuda = torch.cuda.is_available()
        self.WaterAtValue = 0.0192
        self.root_path = '/home/asus/Documents/lh/Datasets/Training_Images/1mm_B30/full_1mm'
        self.target_folder = 'Results/'

        self.reduction = 16
        
        self.batch_num = {'train': 10000, 'val': 1000, 'test': 10000}
        self.batch_size = {'train': 1, 'val': 1, 'test': 1}
        self.num_workers = {'train': 20, 'val': 10, 'test': 10}
        self.reload_mode = 'train'
        self.scale_factor = 2 # if the scale_factor changed, the network architecture should be changed accordingly.
        self.num_filters = 16
        self.is_train = True  # running the training phase if is true, otherwise, the testing phase.
        self.reload_model = False
        self.is_lr_scheduler = False
        self.geo_mode = 'fanflat' # or 'parallel'
        self.angle_range = {'fanflat': 2*np.pi, 'parallel': np.pi}
        self.net_name = 'test'
        self.gpu_id_conv = 0
        self.gpu_id_bp = 0
        self.save_as_mat = True
        # self.is_shuffle = True
        self.is_shuffle = True if self.is_train else False

        if self.Dataset_name in ['Mayo']:
            self.TrainFolder = {'patients': ['L096','L109','L143','L192','L286','L291','L310','L333', 'L506'], 'HighDose': ['full'], 'LowDose': ['quarter'], 'SliceThickness': ['1mm']}
            self.ValFolder = {'patients': ['L067'], 'HighDose': ['full'], 'LowDose': ['quarter'], 'SliceThickness': ['1mm']}
            self.TestFolder = {'patients': ['L067'], 'HighDose': ['full'], 'LowDose': ['quarter'], 'SliceThickness': ['1mm']}
        elif self.Dataset_name in ['MayoRaw']:
            self.TrainFolder = {'patients': ['L096','L109','L143','L192','L286','L291','L310','L333', 'L506'], 'HighDose': ['1dose'], 'LowDose': ['0.25dose'], 'SliceThickness': ['1mm']}
            # self.TrainFolder = {'patients': ['L067'], 'HighDose': ['1dose'], 'LowDose': ['0.25dose'], 'SliceThickness': ['1mm']}
            self.ValFolder = {'patients': ['L067'], 'HighDose': ['1dose'], 'LowDose': ['0.25dose'], 'SliceThickness': ['1mm']}
            self.TestFolder = {'patients': ['L067'], 'HighDose': ['1dose'], 'LowDose': ['0.25dose'], 'SliceThickness': ['1mm']}
        
        # self.result_folder = self.net_name + '_result'
        # if not os.path.isdir(self.target_folder + '/' + self.result_folder):
        #     os.makedirs(self.target_folder + '/' + self.result_folder)
            
        # for save_folder in ['Model_save', 'Loss_save', 'Optimizer_save']:
        #     if not os.path.isdir(self.target_folder + save_folder):
        #         os.makedirs(self.target_folder + save_folder)
            