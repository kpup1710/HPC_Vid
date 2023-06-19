from typing import Any
import numpy as np
import pickle
import torch
import json
import random
import itertools
import math
import pandas as pd
import sys
import pdb
import os
from softdtw import SoftDTW
from utils import dct_2d
from torch.utils.data import Dataset



class Feeder(Dataset):
    def __init__(self, data_path, split='train', p_interval=None, repeat=5, random_choose=False, random_shift=False, random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True, bone=False, vel=False, sort=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        self.debug = debug
        self.data_path = data_path
        # self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.vel = vel
        self.time_steps = 40
        self.bone = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),(8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (1, 15), (1, 16), (1, 17), (1, 18), (11, 24), (14, 21), (14, 19), (14, 20), (11, 22), (11, 23)]
        self.load_data(self.data_path)

    def load_data(self, data_path):
        # data: N C V T M
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if self.split == 'train':
            data_dict = data['train']
        elif self.split == 'test':
            data_dict =  data['test']

        self.data = []
        self.labels = []
        for k, v in data_dict.items():
            value = np.array(v)
            lb = k[-1]
            T, C, N = value.shape
            value =  np.reshape(value,(T,N,C))
            self.data.append(value)
            self.labels.append(lb)
            
    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index: Any) -> Any:
        label = self.labels[index]
        value = self.data[index]
        if self.split == 'train':
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1
            scalerValue = np.reshape(scalerValue, (-1, 25, 3))

            data = np.zeros( (self.time_steps, 25, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length))*100, self.time_steps)
            random_idx.sort()
            data[:,:,:] = value[random_idx,:,:]
            data[:,:,:] = value[random_idx,:,:]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1

            scalerValue = np.reshape(scalerValue, (-1, 25, 3))

            data = np.zeros( (self.time_steps, 25, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            idx = np.linspace(0,length-1,self.time_steps).astype(int)
            data[:,:,:] = value[idx,:,:] # T,V,C
        data = np.transpose(data, (2, 0, 1))
        C,T,V = data.shape
        data = np.reshape(data,(C,T,V,1))
        return data, label, index
    
    def __len__(self):
        return len(self.data)
    
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == '__main__':
    dataset = Feeder(data_path='C:\\Users\\RedmiBook\\HUST\\Documents\\Studying\\Deep Learning\\project\\Human Pose Estimation\\code\\HPC_vid\\HPC_Vid\\data\\ec3d\\ec3d.pickle')
    print(dataset[0][0].shape)
