def noDIA_feat_oneepoch(data, device):
    srch, tmpl, features, labels, ids = data
    test_data = torch.cat([srch, tmpl], 3)
    test_data = test_data.to(device)
    features = features.to(device)
    labels = labels.to(device)
    return test_data, labels, features

def DIA_oneepoch(data, device):
    diff, srch, tmpl, labels, ids = data
    train_data = torch.cat([diff, srch, tmpl], 3)
    train_data = train_data.to(device)
    labels = labels.to(device)
    return train_data, labels

def noDIA_oneepoch(data, device):
    srch, tmpl, labels, ids = data
    train_data = torch.cat([srch, tmpl], 3)
    train_data = train_data.to(device)
    labels = labels.to(device)
    return train_data, labels

import DataSetLoad as data_load
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, TensorDataset, WeightedRandomSampler
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torchvision import transforms, utils
from torchvision.transforms import functional as F
import glob
import numpy as np
import pandas as pd
import torch
import random

def create_noDIA_norm_feat_dataset(batch_size, size=20, split='train', scaler=RobustScaler()):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/data_split_3s/")
    csv_loc = os.path.join(ROOT, "data/autoscan_features.3.feather")
    np.random.seed(65)
    
    if split == 'train':
        folder = 'individual_train/*'
    else:
        folder = 'individual_test/*'
    test_loc = np.random.choice(glob.glob(data_loc+folder), size=size, replace=False)
    dataset_test = data_load.NpyDataset_triplet_to_ind_triplet_feat(test_loc, csv_loc, scaler)

    labels = [int(iu.split("/")[-1].split("_")[-1][0]) for iu in test_loc]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    seed = 42
    torch.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=size, replacement=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, sampler=sampler)
    
    return test_dataloader

def create_noDIA_nonorm_feat_dataset(batch_size, size=20, split='train', scaler=RobustScaler()):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/data_split/")
    csv_loc = os.path.join(ROOT, "data/autoscan_features.3.feather")
    np.random.seed(65)
    torch.manual_seed(8)
    
    if split == 'train':
        folder = 'individual_train'
    else:
        folder = 'individual_test'
        
    train_loc = os.path.join(data_loc+folder)
    train_loc_all = glob.glob(data_loc+folder+'/tmpl*')
    dataset_train = data_load.NpyDataset_noDIACNN_features(train_loc, csv_loc, scaler)

    labels = [int(iu.split("/")[-1].split("_")[-1][0]) for iu in train_loc_all]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=size, replacement=True)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=4)
    return train_dataloader

def create_DIA_nonorm_dataset(batch_size, size=20, split='train'):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/data_split/")
    os.environ['PYTHONHASHSEED'] = str(76)
    random.seed(45)
    np.random.seed(65)
    torch.manual_seed(8)
    if split == 'train':
        folder = 'individual_train'
    else:
        folder = 'individual_test'
        
    train_loc = os.path.join(data_loc+folder)
    train_loc_all = glob.glob(data_loc+folder+'/tmpl*')
    print(data_loc, train_loc)
    dataset_train = data_load.NpyDataset_ind_triplet(train_loc) 
    labels = [int(iu.split("/")[-1].split("_")[-1][0]) for iu in train_loc_all]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=size, replacement=True)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=4)
    return train_dataloader

def create_noDIA_nonorm_dataset(batch_size, size=20, split='train'):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/data_split/")
    os.environ['PYTHONHASHSEED'] = str(76)
    random.seed(45)
    np.random.seed(65)
    torch.manual_seed(8)
    if split == 'train':
        folder = 'individual_train'
    else:
        folder = 'individual_test'
        
    train_loc = os.path.join(data_loc+folder)
    train_loc_all = glob.glob(data_loc+folder+'/tmpl*')
    print(data_loc, train_loc)
    dataset_train = data_load.NpyDataset_ind_triplet_noDIA(train_loc) 
    labels = [int(iu.split("/")[-1].split("_")[-1][0]) for iu in train_loc_all]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=size, replacement=True)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=4)
    return train_dataloader


def create_DIA_norm_dataset(batch_size, size=20, split='train'):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/data_split_3s/")
    if split == 'train':
        folder = 'individual_train/*'
    else:
        folder = 'individual_test/*'

    np.random.seed(65)
    train_loc = np.random.choice(glob.glob(data_loc+folder), size=size, replace=False)
    dataset_train = data_load.NpyDataset_triplet_to_ind_triplet(train_loc)

    labels = [int(iu.split("/")[-1].split("_")[-1][0]) for iu in train_loc]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    seed = 8
    torch.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
    return train_dataloader


def create_noDIA_norm_dataset(batch_size, size=20, split='train'):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/data_split_3s/")
    if split == 'train':
        folder = 'individual_train/*'
    else:
        folder = 'individual_test/*'

    np.random.seed(65)
    train_loc = np.random.choice(glob.glob(data_loc+folder), size=size, replace=False)
    dataset_train = data_load.NpyDataset_triplet_to_ind_triplet_noDIA(train_loc)

    labels = [int(iu.split("/")[-1].split("_")[-1][0]) for iu in train_loc]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    seed = 8
    torch.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
    return train_dataloader

def create_DIA_norm_dc2_dataset(batch_size = 128, split='train'):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print(ROOT)
    data_loc = os.path.join(ROOT, "data/raw_npy")
    labels_csv = os.path.join(ROOT, "data/"+"sources_with_labels.csv")
    
    class SameRotationTransform(object):
        def __init__(self, out_angle):
            self.angle = out_angle

        def __call__(self, img):
            return F.rotate(img[0], self.angle), F.rotate(img[1], self.angle), F.rotate(img[2], self.angle)
    
    class RotationTriple(object):
        def __init__(self, out_size):
            self.size = out_size

        def __call__(self, img):
            return F.resize(img[0], size=self.size), F.resize(img[1], size=self.size), F.resize(img[2], size=self.size)

    class HFlipTriple(object): 
        def __call__(self, img):
            return F.hflip(img[0]), F.hflip(img[1]), F.hflip(img[2])

    class VFlipTriple(object): 
        def __call__(self, img):
            return F.vflip(img[0]), F.vflip(img[1]), F.vflip(img[2])
        
    composed = transforms.Compose([RotationTriple([51,51]),
                               SameRotationTransform(90),
                               SameRotationTransform(180),
                               SameRotationTransform(270),
                               HFlipTriple(),
                               VFlipTriple()
                               ])

    labels_csv_df = pd.read_csv(labels_csv)
    test_split = 0.1
    num_data = len(labels_csv_df)
    num_test = int(test_split * num_data)
    num_valid = int(test_split * num_data)
    num_train = num_data - num_test - num_valid
    seed = 1
    indices = list(range(num_data))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, test_indices, valid_indices = indices[:num_train], indices[num_train:num_train+num_test], indices[num_train+num_test:num_train+num_test+num_valid]

    
    def weights(indices):
        labels = labels_csv_df['real'][indices]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return sample_weights
    
    
    dataset_train = data_load.NpyDataset_ind_triplet_dc2(data_loc, labels_csv, split='train', transform_real=composed, transform_bogus=RotationTriple([51,51]))
    seed = 8
    torch.manual_seed(seed)
    sample_weights = weights(train_indices)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(dataset_train, sampler=train_sampler, batch_size=batch_size)

    dataset_test = data_load.NpyDataset_ind_triplet_dc2(data_loc, labels_csv, split='test', transform_real=composed, transform_bogus=RotationTriple([51,51]))
    seed = 8
    torch.manual_seed(seed)
    sample_weights = weights(test_indices)
    test_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    test_dataloader = DataLoader(dataset_test, sampler=test_sampler, batch_size=batch_size)
    
    dataset_valid = data_load.NpyDataset_ind_triplet_dc2(data_loc, labels_csv, split='valid', transform_real=composed, transform_bogus=RotationTriple([51,51]))
    seed = 8
    torch.manual_seed(seed)
    sample_weights = weights(valid_indices)
    valid_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    valid_dataloader = DataLoader(dataset_valid, sampler=valid_sampler, batch_size=batch_size)
    
    if split == 'train':
        return train_dataloader
    elif split == 'test':
        return test_dataloader
    elif split == 'valid':
        return valid_dataloader