import numpy as np
import os
import os.path as osp
import scipy.io as scio
import h5py
import copy
import torch.utils.data as D

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def train_val_dataset(config, dataset, val_split=0.3):
    
    # stratified sampling
    train_idx, val_idx = train_test_split(list(range(len(dataset))), 
                                          test_size=val_split, 
                                          random_state=config.seed, 
                                          stratify=dataset.label,
                                          )
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


class CortexDataset(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, rootdir, tasks=None, transform=None, mode=None):
        self.transform = transform
        
        # data dir
        if mode is not None:
            datadir = osp.join(rootdir, mode+'.mat')
        else:
            datadir = osp.join(rootdir, 'data.mat')
        
        idxdir = osp.join(rootdir, 'idx_surf.mat')
        
        # load mat v7.3 files
        tmp_data = h5py.File(datadir, 'r')
        data = tmp_data['data'][:, :].transpose()
        label = tmp_data['label'][:, :].transpose().astype(np.int64)
        
        # idx of surface
        if osp.exists(idxdir):
            tmp_idx = h5py.File(idxdir, 'r')
            idx_surf = tmp_idx['idx_data'][:, :].transpose().astype(np.int64)
            
            # Because the index in the mat file begins at 1, but in the vtk 
            # file, it starts at 0, so it should minus 1 for our later 
            # processing
            idx_surf -= 1
        else:
            idx_surf = None
        
        # tasks: 
        #    'three_classes': three-classes task (sulc, 2HGs and 3HGs)
        #    'hinges': two-classes gyral task (2HGs and 3HGs)
        #    'sulc_gyral': two-classes sulcal and gyral task (sulci and gyri)
        
        if tasks != 'three_classes':
            if tasks == 'hinges':
                ind_2HGs = np.where(label == 1)[0]
                ind_3HGs = np.where(label == 2)[0]
                
                data_2HGs = data[ind_2HGs]
                data_3HGs = data[ind_3HGs]
                data = np.vstack((data_2HGs, data_3HGs))
                
                label_2HGs = label[ind_2HGs]
                label_3HGs = label[ind_3HGs]
                # Note: one-hot encoding labels must begin 0
                label = np.vstack((label_2HGs, label_3HGs)) - 1
                
                if idx_surf is not None:
                    idx_2HGs = idx_surf[ind_2HGs]
                    idx_3HGs = idx_surf[ind_3HGs]
                    idx_surf = np.vstack((idx_2HGs, idx_3HGs))
                
            elif tasks == 'sulc_gyral':
                # merge the labels of 2HGs and 3HGs into gyral signals
                label[label == 2] = 1
        
        # add idx_surf in the first column of data
        if idx_surf is not None:
            self.data = np.hstack((idx_surf, data))
        else:
            self.data = data
        self.label = label
        
        # sample size
        self.len = len(self.data)
        
        # signal length, which is mainly used MobileNetV2_1d
        self.input_size = self.data.shape[1]
        
        # set data channel number
        if self.data.ndim > 2:
            self.datadim = self.data.ndim
        else:
            self.datadim = 1
        
        # classes
        self.classes = len(np.unique(self.label))
        
        # class names
        # self.labelname = ['Sulcal', '2HGs', '3HGs']
        
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        """
        data = self.data[index, :]
        # reshape to [Channel x Length]
        data = data.reshape((self.datadim, -1))
        label = self.label[index].astype('int64')
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def equal_list(list):
    """
        Judge if all the elements in a list are equal.
    """
    
    if all(elem == list[0] for elem in list):
        return True
    else:
        return False


def find_largest_list(list):
    """
        Find the largest value in a list.
    """
    
    list_new = copy.deepcopy(list)
    list_new.sort()
    return list_new[-1]


def merge_data(path, subids, group='patient', mode='train'):
    """
        Merge the Matfile based on the specified subids.
    """
    
    merged_mat = [
            scio.loadmat(osp.join(path, str(x), mode + '.mat')) for x in subids]
    merged_data = [x['data'] for x in merged_mat]
    merged_label = [x['label'] for x in merged_mat]
    
    merged_size = [x.shape[1] for x in merged_data]
    if not equal_list(merged_size):    
        merged_largest = find_largest_list(merged_size)
        
        for i, size in enumerate(merged_size):
            if not size == merged_largest:
                temp = merged_data[i]
                pad_zeros = np.zeros((temp.shape[0], merged_largest - size))
                temp_new = (temp, pad_zeros)
                merged_data[i] = np.hstack(temp_new)
    
    merged_data_all = np.concatenate(merged_data)
    merged_label_all = np.vstack(merged_label)
    merged_all = {'data': merged_data_all, 'label': merged_label_all}
    
    path_des = osp.join(path, group)
    if not osp.isdir(path_des):
        os.makedirs(path_des)
    
    scio.savemat(osp.join(path_des, mode), merged_all)






