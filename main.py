#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time
import glob
import pickle
import random
import traceback
import resource
import torch.nn.functional as F

from collections import OrderedDict
from feeders.feeder_corr_ec3d import Feeder as Cor_Feeder

# import apex
import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
# from sklearn.metrics import confusion_matrix

from args import get_parser
from loss import LabelSmoothingCrossEntropy, get_mmd_loss
from model.infogcn import InfoGCN
from model.model import Predictor_Corrector
from utils import get_vector_property
from utils import BalancedSampler as BS
from loss import SoftDTW, dtw_loss
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda()
        # self.model = torch.nn.DataParallel(model, device_ids=(0,1,2))

    def load_data(self):
        Pred_Feeder = import_class(self.arg.feeder)
        self.pred_data_loader = dict()
        self.cor_data_loader = dict()
        data_path = f'{self.arg.root_path}/{self.arg.dataset}.pickle'
        if self.arg.phase == 'train':
            dt = Pred_Feeder(data_path=data_path,
                split='train',
                window_size=64,
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=True if self.arg.balanced_sampling else False,
            )
            cor_dt = Cor_Feeder(data_path=f'{self.arg.root_path}/corr_ec3d.pickle', split='train',
                                p_interval=[0.5, 1],
                                vel=self.arg.use_vel,
                                random_rot=self.arg.random_rot,
                                sort=True if self.arg.balanced_sampling else False,)
            if self.arg.balanced_sampling:
                sampler = BS(data_source=dt, args=self.arg)
                cor_sampler = BS(data_source=cor_dt, args=self.arg)
                shuffle = False
            else:
                sampler = None
                cor_sampler = None
                shuffle = True

            self.pred_data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dt,
                sampler=sampler,
                batch_size=self.arg.batch_size,
                shuffle=shuffle,
                num_workers=self.arg.num_worker,
                drop_last=True,
                pin_memory=True,
                worker_init_fn=init_seed)

            
            self.cor_data_loader['train'] = torch.utils.data.DataLoader(
                dataset=cor_dt,
                sampler=cor_sampler,
                batch_size=self.arg.batch_size,
                shuffle=shuffle,
                num_workers=self.arg.num_worker,
                drop_last=True,
                pin_memory=True,
                worker_init_fn=init_seed)
            
        self.pred_data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Pred_Feeder(
                data_path=data_path,
                split='test',
                window_size=64,
                p_interval=[0.95],
                vel=self.arg.use_vel
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)
        
        self.cor_data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Cor_Feeder(
                data_path=f'{self.arg.root_path}/corr_ec3d.pickle',
                split='test',
                p_interval=[0.95],
                vel=self.arg.use_vel
            ),
            batch_size=1,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)

    def load_model(self):
        self.model = Predictor_Corrector(args=self.arg).float()

        # x = torch.randn((64, 3, 42, 25,1))
        # y = self.model(x)

        self.loss = LabelSmoothingCrossEntropy().cuda()
        self.cor_loss = F.cross_entropy

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda()] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.predictor.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            
            self.cor_optimizer = optim.SGD(
                self.model.corrector.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.predictor.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

            self.cor_optimizer = optim.Adam(
                self.model.corrector.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def adjust_learning_rate(self, epoch, phase):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch and self.arg.weights is None:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            if phase == 'pred':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
            else:
                for param_group in self.cor_optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train_predictor(self, epoch):
        self.model.predictor.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.adjust_learning_rate(epoch, phase='pred')

        loss_value = []
        mmd_loss_value = []
        l2_z_mean_value = []
        acc_value = []
        cos_z_value = []
        dis_z_value = []
        cos_z_prior_value = []
        dis_z_prior_value = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        for data, y, index in tqdm(self.pred_data_loader['train'], dynamic_ncols=True):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda()
                y = y.long().cuda()
            timer['dataloader'] += self.split_time()

            # forward
            y_hat, z = self.model.predictor(data.float())
            mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.predictor.z_prior, y, self.arg.num_class)
            cos_z, dis_z = get_vector_property(z_mean)
            cos_z_prior, dis_z_prior = get_vector_property(self.model.predictor.z_prior)
            cos_z_value.append(cos_z.data.item())
            dis_z_value.append(dis_z.data.item())
            cos_z_prior_value.append(cos_z_prior.data.item())
            dis_z_prior_value.append(dis_z_prior.data.item())

            cls_loss = self.loss(y_hat, y)
            loss = self.arg.lambda_2* mmd_loss + self.arg.lambda_1* l2_z_mean + cls_loss
            # backward
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            loss_value.append(cls_loss.data.item())
            mmd_loss_value.append(mmd_loss.data.item())
            l2_z_mean_value.append(l2_z_mean.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(y_hat.data, 1)
            acc = torch.mean((predict_label == y.data).float())
            acc_value.append(acc.data.item())

            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(f'\tTraining loss: {np.mean(loss_value):.4f}.  Training acc: {np.mean(acc_value)*100:.2f}%.')
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

    def eval_predictor(self, epoch, save_score=False, loader_name=['test'], save_z=False, save_model=False):
        self.model.predictor.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            mmd_loss_value = []
            l2_z_mean_value = []
            score_frag = []
            label_list = []
            pred_list = []
            cos_z_value = []
            dis_z_value = []
            cos_z_prior_value = []
            dis_z_prior_value = []
            step = 0
            z_list = []
            for data, y, index in tqdm(self.pred_data_loader[ln], dynamic_ncols=True):
                label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    y = y.long().cuda()
                    y_hat, z = self.model.predictor(data)
                    if save_z:
                        z_list.append(z.data.cpu().numpy())
                    mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.predictor.z_prior, y, self.arg.num_class)
                    cos_z, dis_z = get_vector_property(z_mean)
                    cos_z_prior, dis_z_prior = get_vector_property(self.model.predictor.z_prior)
                    cos_z_value.append(cos_z.data.item())
                    dis_z_value.append(dis_z.data.item())
                    cos_z_prior_value.append(cos_z_prior.data.item())
                    dis_z_prior_value.append(dis_z_prior.data.item())
                    cls_loss = self.loss(y_hat, y)
                    loss = self.arg.lambda_2*mmd_loss + self.arg.lambda_1*l2_z_mean + cls_loss
                    score_frag.append(y_hat.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    cls_loss_value.append(cls_loss.data.item())
                    mmd_loss_value.append(mmd_loss.data.item())
                    l2_z_mean_value.append(l2_z_mean.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            cls_loss = np.mean(cls_loss_value)
            mmd_loss = np.mean(mmd_loss_value)
            l2_z_mean_loss = np.mean(l2_z_mean_value)
            if 'ec3d' in self.arg.feeder:
                self.pred_data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.pred_data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, np.mean(cls_loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.pred_data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            accuracy = self.pred_data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                with open(f'{self.arg.work_dir}/best_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

                if save_model:
                    state_dict = self.model.predictor.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
    
                    torch.save(weights, f'{self.arg.work_dir}/best_pred.pt')


            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)

            if save_z:
                z_list = np.concatenate(z_list)
                np.savez(f'{self.arg.work_dir}/z_values.npz', z=z_list, z_prior=self.model.predictor.z_prior.cpu().numpy(), y=label_list)
    
    def train_corrector(self, epoch):
        self.model.predictor.eval()
        label_list = []
        pred_list = []
        score_frag = []
        for data, y,_, index in tqdm(self.cor_data_loader['test'], dynamic_ncols=True):
                label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    y = y.long().cuda()
                    y_hat, z = self.model.predictor(data)
                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    score_frag.append(y_hat.data.cpu().numpy())

        score = np.concatenate(score_frag)
        accuracy = self.cor_data_loader['test'].dataset.top_k(score, 1)
        print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
        # acc for each class:
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)

        self.model.corrector.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.adjust_learning_rate(epoch, phase='cor')
        crit = SoftDTW(use_cuda=True)
        loss_value = []
        dwt_loss_value = []
        ce_loss_value = []
        acc_value = []
        
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        for data, y,_, index in tqdm(self.cor_data_loader['train'], dynamic_ncols=True):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda()
                y = y.long().cuda()
            timer['dataloader'] += self.split_time()
            # forward
            _, y_hat_cor, x_cor = self.model(data.float())
            # print(x_cor.shape)
            # print(data.shape)
            corr_loss,_ = dtw_loss(x_cor, data, crit, is_cuda=True)
            corr_loss = corr_loss/data.shape[0]
            cls_loss = self.cor_loss(y_hat_cor, y)
            loss = 0.7* corr_loss + 0.3* cls_loss
            # backward
            self.cor_optimizer.zero_grad()
            loss.backward()
            self.cor_optimizer.step()
            loss_value.append(loss.data.item())
            dwt_loss_value.append(corr_loss.data.item())
            ce_loss_value.append(cls_loss.data.item())
            timer['model'] += self.split_time()
            value, predict_label = torch.max(y_hat_cor.data, 1)
            acc = torch.mean((predict_label == y.data).float())
            acc_value.append(acc.data.item())
            timer['statistics'] += self.split_time()
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(f'\tTraining loss: {np.mean(loss_value):.4f}.  Training acc: {np.mean(acc_value)*100:.2f}%.')
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

    
    def eval_corrector(self, epoch,save_score, loader_name, save_model):
        self.model.corrector.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            dwt_loss_value = []
            label_loss_value = []
            ce_loss_value = []
            acc_value = []
            label_list = []
            pred_list = []
            score_frag = []
            crit = SoftDTW(use_cuda=True)

            for data, y,cor_label, index in tqdm(self.cor_data_loader[ln], dynamic_ncols=True):
                label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    y = y.long().cuda()
                    # forward
                    _, y_hat_cor, x_cor = self.model(data.float())
                    # print(x_cor.shape)
                    # print(data.shape)
                    corr_loss,_ = dtw_loss(x_cor, data, crit, is_cuda=True)
                    corr_label_loss, _ = dtw_loss(x_cor, cor_label, crit, is_cuda=True)
                    corr_loss = corr_loss/data.shape[0]
                    corr_label_loss = corr_label_loss/data.shape[0]
                    cls_loss = self.cor_loss(y_hat_cor, y)
                    loss = 0.7* corr_loss + 0.3* cls_loss
    
                    _, predict_label = torch.max(y_hat_cor.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
    
                    score_frag.append(y_hat_cor.data.cpu().numpy())
    
                    loss_value.append(loss.data.item())
    
                    dwt_loss_value.append(corr_loss.data.item())
                    label_loss_value.append(corr_label_loss.item())
                    ce_loss_value.append(cls_loss.data.item())
   


            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            dwt_loss = np.mean(dwt_loss_value)
            label_loss = np.mean(label_loss_value)
            ce_loss = np.mean(ce_loss_value)
            if 'ec3d' in self.arg.feeder:
                self.cor_data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.cor_data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, np.mean(ce_loss_value)))
            self.print_log('\tMean {} label-loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, label_loss))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.cor_data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_cor_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            accuracy = self.cor_data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                with open(f'{self.arg.work_dir}/best_cor_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

                if save_model:
                    state_dict = self.model.corrector.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
    
                    torch.save(weights, f'{self.arg.work_dir}/best_cor.pt')


            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)

    def get_pose_list(self, pred_path, cor_path):
        self.model.load_predictor(path=pred_path)
        self.model.load_corrector(path=cor_path)
        self.model.eval()
        pose_list = []
        label_list = []

        for data, y,cor_label, index in tqdm(self.cor_data_loader['test'], dynamic_ncols=True):
                # label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    print(cor_label.shape)
                    y = y.long().cuda()
                    # forward
                    _, y_hat_cor, x_cor = self.model(data.float())
                    pose_list.append(x_cor.cpu().numpy())
                    label_list.append(cor_label.cpu().numpy())
        
        with open(f'{self.arg.work_dir}/cor_pose.pkl', 'wb') as f:
                    pickle.dump(pose_list, f)

        with open(f'{self.arg.work_dir}/test_pose.pkl', 'wb') as f:
                    pickle.dump(label_list, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = 0
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters Predictor: {count_parameters(self.model.predictor)}')
            self.print_log(f'Start training Predictor')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # save_model = (epoch + 1 == self.arg.num_epoch) and (epoch + 1 > self.arg.save_epoch)

                self.train_predictor(epoch)

                # if epoch > 80:
                self.eval_predictor(epoch, save_score=self.arg.save_score, loader_name=['test'], save_model=True)

            # test the best model
            print(glob.glob(os.path.join(self.arg.work_dir, 'best_pred'+'*')))
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'best_pred'+'*'))[0]
            
            # weights = torch.load(weights_path)
            # self.model.predictor.load_state_dict(weights)
            self.model.load_predictor(path=weights_path)

            self.arg.print_log = False
            self.eval_predictor(epoch=0, save_score=True, loader_name=['test'])
            self.arg.print_log = True


            # num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            # self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')



            self.print_log(f'Start training Corrector')
            self.global_step = 0
            self.best_acc = 0
            self.print_log(f'# Parameters Corrector: {count_parameters(self.model.corrector)}')
            for epoch in range(0, 150):
                # save_model = (epoch + 1 == self.arg.num_epoch) and (epoch + 1 > self.arg.save_epoch)

                self.train_corrector(epoch)

                # if epoch > 80:
                self.eval_corrector(epoch, save_score=self.arg.save_score, loader_name=['test'], save_model=True)
        
            # num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            # self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            pred_path = f'{self.arg.work_dir}/best_pred.pt'
            cor_path = f'{self.arg.work_dir}/best_cor.pt'
            self.get_pose_list(pred_path=pred_path, cor_path=cor_path)
            self.print_log('Done.\n')

def main():
    # parser arguments
    parser = get_parser()
    arg = parser.parse_args()
    arg.work_dir = f"results/{arg.dataset}_{arg.datacase}"
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
    print("finished")
