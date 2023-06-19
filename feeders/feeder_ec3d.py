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
from softdtw import SoftDTW
from utils import dct_2d
from torch.utils.data import Dataset



class Feeder(Dataset):
    def __init__(self, data_path, split=None, p_interval=None, repeat=5, random_choose=False, random_shift=False, random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True, bone=False, vel=False, sort=False)
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
        self.label_path = label_path
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
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
    
    def load_data(self):
        # data: N C V T M
        self.data = []
        for data in self.data_dict:
            file_name = data['file_name']
            with open(self.nw_ucla_root + file_name + '.json', 'r') as f:
                json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons)
            self.data.append(value)


class EC3D(Dataset):
    def __init__(self, data_path, dct_n=25, split=0, sets=None, is_cuda=False, add_data=None):
        if sets is None:
            sets = [[0, 1], [2], [3]]
        self.dct_n = dct_n
        correct, other = load_data(data_path, sets[split], add_data=add_data)
        pairs = dtw_pairs(correct, other, is_cuda=is_cuda)

        self.targets_label = [i[1] for i in pairs]
        self.inputs_label = [i[0] for i in pairs] 

        self.targets = [correct[i] for i in self.targets_label]
        self.inputs_raw = [other[i] for i in self.inputs_label]
        
        self.inputs = [dct_2d(torch.from_numpy(x))[:, :self.dct_n].numpy() if x.shape[1] >= self.dct_n else
                       dct_2d(torch.nn.ZeroPad2d((0, self.dct_n - x.shape[1], 0, 0))(torch.from_numpy(x))).numpy()
                       for x in self.inputs_raw]

        self.node_n = np.shape(self.inputs_raw[0])[0]
        self.batch_ids = list(range(len(self.inputs_raw)))
        self.name = "EC3D"
        # pdb.set_trace()
        # with open('data/DTW_Method.pickle', 'wb') as f:
        #     pickle.dump({'targets':self.targets,'tar_label':self.targets_label,'inputs':self.inputs,'inputs_raw':self.inputs_raw, 'inputs_label': self.inputs_label}, f)


    def __len__(self):
        return np.shape(self.inputs)[0]

    def __getitem__(self, item):
        return self.batch_ids[item], self.inputs[item]

def load_data(data_path, subs, add_data=None):
    with open(data_path, "rb") as f:
        data_gt = pickle.load(f)

    if add_data is not None:
        with open(add_data, "rb") as f:
            data = pickle.load(f)
        labels = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'cam'])
    else:
        data = data_gt
        labels = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
        labels['cam'] = 'gt'
    # import pdb; pdb.set_trace()
    joints = list(range(15)) + [19, 21, 22, 24]

    labels_gt = pd.DataFrame(data_gt['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
    labels_gt['cam'] = 'gt'

    labels[['lab', 'rep']] = labels[['lab', 'rep']].astype(int)
    labels_gt[['lab', 'rep']] = labels_gt[['lab', 'rep']].astype(int)

    subs = labels[['act', 'sub', 'lab', 'rep']].drop_duplicates().groupby('sub').count().rep[subs]

    indices = labels['sub'].isin(subs.index)
    indices_gt = labels_gt['sub'].isin(subs.index)
    labels = labels[indices]
    labels_gt = labels_gt[indices_gt]

    lab1 = labels_gt[labels_gt['lab'] == 1].groupby(['act', 'sub', 'lab', 'rep', 'cam']).groups
    labnot1 = labels.groupby(['act', 'sub', 'lab', 'rep', 'cam']).groups

    poses = data['poses'][:, :, joints]
    poses_gt = data_gt['poses'][:, :, joints]

    correct = {k: poses_gt[v].reshape(-1, poses_gt.shape[1] * poses_gt.shape[2]).T for k, v in lab1.items()}
    other = {k: poses[v].reshape(-1, poses.shape[1] * poses.shape[2]).T for k, v in labnot1.items()}

    return correct, other


def dtw_pairs(correct, incorrect, is_cuda=False):
    pairs = []
    for act, sub in set([(k[0], k[1]) for k in incorrect.keys()]):
        ''' fetch from all sets or only training set (dataset_fetch baseline used to compare dtw_loss)'''
        correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] == sub} # all dastasets
        # correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] != sub}  # training sets
        incorrect_sub = {k: v for k, v in incorrect.items() if k[0] == act and k[1] == sub}
        dtw_sub = {k: {} for k in incorrect_sub.keys()}
        for i, pair in enumerate(itertools.product(incorrect_sub, correct_sub)):
            criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
            if is_cuda:
                p0 = torch.from_numpy(np.expand_dims(incorrect_sub[pair[0]].T, axis=0)).cuda()
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0)).cuda()
            else:
                p0 = torch.from_numpy(np.expand_dims(incorrect_sub[pair[0]].T, axis=0))
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0))
            dtw_sub[pair[0]][pair[1]] = (criterion(p0, p1) - 1 / 2 * (criterion(p0, p0) + criterion(p1, p1))).item()
        dtw = pd.DataFrame.from_dict(dtw_sub, orient='index').idxmin(axis=1)
        pairs = pairs + list(zip(dtw.index, dtw))
    return pairs

 
def dtw_pairs_4targ(correct, incorrect, is_cuda=False, test=False):
    pairs = []
    for sub in set([k[1] for k in correct.keys()]):
        dtw_sub = {k: {} for k in incorrect.keys()}
        if test:
            correct_sub = correct
        else:
            correct_sub = {k: v for k, v in correct.items() if k[1] == sub}
        for i, pair in enumerate(itertools.product(incorrect, correct_sub)):
            criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
            if is_cuda:
                p0 = torch.from_numpy(np.expand_dims(incorrect[pair[0]].T, axis=0)).cuda()
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0)).cuda()
            else:
                p0 = torch.from_numpy(np.expand_dims(incorrect[pair[0]].T, axis=0))
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0))
            dtw_sub[pair[0]][pair[1]] = (criterion(p0, p1) - 1 / 2 * (criterion(p0, p0) + criterion(p1, p1))).item()
        dtw = pd.DataFrame.from_dict(dtw_sub, orient='index').idxmin(axis=1)
        pairs = pairs + list(zip(dtw.index, dtw))
        if test:
            return pairs
    return pairs

# data_train = HV3D('Data/data_3D.pickle', sets=[[0,1,2],[3]], split=0, is_cuda=True)
# data_test = HV3D('Data/data_3D.pickle', sets=[[0,1],[2],[3]], split=2, is_cuda=True)
# pdb.set_trace()
