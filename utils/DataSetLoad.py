from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, TensorDataset, WeightedRandomSampler
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch

class NpyDataset(Dataset):
    def __init__(self, data_loc):
        self.npy_files = data_loc

    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.npy_files)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')
        helper = self.npy_files[idx].split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids)
        data = torch.tensor(data).type(torch.FloatTensor).unsqueeze(0)
        return data, labels, ids
    
    
class NpyDataset_ind_triplet(Dataset):
    def __init__(self, data_loc):
        self.npy_files = data_loc
        self.file_list = []
        observations = dict()
        for root, subdirs, fns in os.walk(data_loc):
            for fn in fns:
                if fn.endswith('.npy'):
                    obs_num = "".join([x for x in fn if x.isdigit()])
                    observations.setdefault(obs_num, []).append(os.path.join(root, fn))
        for i, (k, v) in enumerate(observations.items()):
            self.file_list.append(v)

    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_paths = self.file_list[idx]
        diff, srch, tmpl, label, ids = self.process_files(file_paths)
        return diff, srch, tmpl, label, ids


    def process_files(self, paths):
        diff_file = [x for x in paths if "diff" in os.path.basename(x)][0]
        search_file = [x for x in paths if "srch" in os.path.basename(x)][0]
        tmpl_file = [x for x in paths if "tmpl" in os.path.basename(x)][0]

        # load and convert to tensor
        diff = np.load(os.path.join(diff_file), mmap_mode='r')
        srch = np.load(os.path.join(search_file), mmap_mode='r')
        tmpl = np.load(os.path.join(tmpl_file), mmap_mode='r')
        
        helper = tmpl_file.split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids)
        diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        
        return diff, srch, tmpl, labels, ids
    
class NpyDataset_ind_triplet_noDIA(Dataset):
    def __init__(self, data_loc):
        self.npy_files = data_loc
        self.file_list = []
        observations = dict()
        for root, subdirs, fns in os.walk(data_loc):
            for fn in fns:
                if fn.endswith('.npy'):
                    obs_num = "".join([x for x in fn if x.isdigit()])
                    observations.setdefault(obs_num, []).append(os.path.join(root, fn))
        for i, (k, v) in enumerate(observations.items()):
            self.file_list.append(v)

    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_paths = self.file_list[idx]
        srch, tmpl, label, ids = self.process_files(file_paths)
        return srch, tmpl, label, ids


    def process_files(self, paths):
        # diff_file = [x for x in paths if "diff" in os.path.basename(x)][0]
        search_file = [x for x in paths if "srch" in os.path.basename(x)][0]
        tmpl_file = [x for x in paths if "tmpl" in os.path.basename(x)][0]

        # load and convert to tensor
        # diff = np.load(os.path.join(diff_file), mmap_mode='r')
        srch = np.load(os.path.join(search_file), mmap_mode='r')
        tmpl = np.load(os.path.join(tmpl_file), mmap_mode='r')
        
        helper = tmpl_file.split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids).type(torch.IntTensor).squeeze()
        # diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        
        return srch, tmpl, labels, ids
    
    
    
class NpyDataset_triplet_to_ind_triplet(Dataset):
    def __init__(self, data_loc):
        self.npy_files = data_loc

    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.npy_files)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        diff = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, :51]
        srch = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, 51:102]
        tmpl = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, 102:]
        
        helper = self.npy_files[idx].split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids)
        diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        return diff, srch, tmpl, labels, ids
    
class NpyDataset_triplet_to_ind_triplet_noDIA(Dataset):
    def __init__(self, data_loc):
        self.npy_files = data_loc

    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.npy_files)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # diff = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, :51]
        srch = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, 51:102]
        tmpl = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, 102:]
        
        helper = self.npy_files[idx].split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids)
        # diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        return srch, tmpl, labels, ids
    
    
class NpyDataset_noDIACNN_features(Dataset):
    def __init__(self, data_loc, features_loc, scaler=RobustScaler()):
        self.npy_files = data_loc
        self.file_list = []
        self.scaler = scaler
        self.feature_file = pd.read_feather(features_loc)
        self.feature_file.loc[:, 'BAND'] = self.feature_file['BAND'].apply(lambda x: self.cat_to_cont(x))
        
        observations = dict()
        for root, subdirs, fns in os.walk(data_loc):
            for fn in fns:
                if fn.endswith('.npy'):
                    obs_num = "".join([x for x in fn if x.isdigit()])
                    observations.setdefault(obs_num, []).append(os.path.join(root, fn))
        for i, (k, v) in enumerate(observations.items()):
            self.file_list.append(v)

    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.file_list)
    
    def cat_to_cont(self, val):
        if val == 'g':
            return 0
        if val == 'i':
            return 1
        if val == 'r':
            return 2 
        if val == 'z':
            return 3

    def __getitem__(self, idx):
        file_paths = self.file_list[idx]
        srch, tmpl, label, ids = self.process_files(file_paths)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.feature_file[self.feature_file["ID"].isin([ids[0].item()])]
        features = features.T.values[2:]
        # np.random.seed(563)
        # features = np.random.rand(features.shape[0], features.shape[1])
        # scaler = RobustScaler()
        features = self.scaler.fit_transform(features.reshape(-1, 1))
        features = np.nan_to_num(features, nan=-1)
        features = torch.tensor(features).type(torch.FloatTensor).squeeze(1)
        return srch, tmpl, features, label, ids
    
    
    def process_files(self, paths):
        # diff_file = [x for x in paths if "diff" in os.path.basename(x)][0]
        search_file = [x for x in paths if "srch" in os.path.basename(x)][0]
        tmpl_file = [x for x in paths if "tmpl" in os.path.basename(x)][0]

        # load and convert to tensor
        # diff = np.load(os.path.join(diff_file), mmap_mode='r')
        srch = np.load(os.path.join(search_file), mmap_mode='r')
        tmpl = np.load(os.path.join(tmpl_file), mmap_mode='r')
        
        helper = tmpl_file.split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids)
        # diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        
        return srch, tmpl, labels, ids
    
class NpyDataset_triplet_to_ind_triplet_feat(Dataset):
    def __init__(self, data_loc, features_loc, scaler=RobustScaler()):
        self.npy_files = data_loc
        self.scaler = scaler
        self.feature_file = pd.read_feather(features_loc)
        self.feature_file.loc[:, 'BAND'] = self.feature_file['BAND'].apply(lambda x: self.cat_to_cont(x))
        
    def __len__(self):
        # Count the total number of samples across all .npy files
        return len(self.npy_files)
    
    def cat_to_cont(self, val):
        if val == 'g':
            return 0
        if val == 'i':
            return 1
        if val == 'r':
            return 2 
        if val == 'z':
            return 3

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # diff = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, :51]
        srch = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, 51:102]
        tmpl = np.load(os.path.join(self.npy_files[idx]), mmap_mode='r')[:, 102:]
        
        helper = self.npy_files[idx].split("/")[-1].split("_")
        labels = [int(helper[-1][0])]
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        ids = [int(helper[1])]
        ids = torch.tensor(ids)
        # diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        
        features = self.feature_file[self.feature_file["ID"].isin([ids[0].item()])]
        features = features.T.values[2:]
        features = self.scaler.fit_transform(features.reshape(-1, 1))
        features = np.nan_to_num(features, nan=-1)
        features = torch.tensor(features).type(torch.FloatTensor).squeeze(1)
        
        return srch, tmpl, features, labels, ids
    
    
    
class NpyDataset_ind_triplet_dc2(Dataset):
    
    def __init__(self, data_loc, features_loc, split=None, transform_real=None, transform_bogus=None):
        self.npy_files = data_loc
        self.feature_file = pd.read_csv(features_loc)
        self.file_list = []
        self.transform_real = transform_real
        self.transform_bogus = transform_bogus
        self.split = split
        observations = dict()
        for root, subdirs, fns in os.walk(data_loc):
            for fn in fns:
                if fn.endswith('.npy'):
                    obs_num = "".join([x for x in fn if x.isdigit()])
                    observations.setdefault(obs_num, []).append(os.path.join(root, fn))
        for i, (k, v) in enumerate(observations.items()):
            self.file_list.append(v)

    def __len__(self):
        test_split = 0.1
        num_data = len(self.file_list)
        num_test = int(test_split * num_data)
        num_valid = int(test_split * num_data)
        num_train = num_data - num_test - num_valid
        seed = 1
        indices = list(range(num_data))
        np.random.seed(seed)
        np.random.shuffle(indices)
        self.train_indices, self.test_indices, self.valid_indices = indices[:num_train], indices[num_train:num_train+num_test], indices[num_train+num_test:num_train+num_test+num_valid]
        if self.split == 'train':
            self.file_list = self.file_list[self.train_indices]
            return len(self.train_indices)
        if self.split == 'valid':
            self.file_list = self.file_list[self.valid_indices]
            return len(self.valid_indices)
        if self.split == 'test':
            self.file_list = self.file_list[self.test_indices]
            return len(self.test_indices)
    
    def __getitem__(self, idx):
        file_paths = self.file_list[idx]
        diff, srch, tmpl, label, ids = self.process_files(file_paths)
        if (label == 0) and self.transform_bogus:
            diff, srch, tmpl = self.transform_bogus([diff, srch, tmpl])
        if (label == 1) and self.transform_real:
            diff, srch, tmpl = self.transform_real([diff, srch, tmpl])

        return diff, srch, tmpl, label, ids


    def process_files(self, paths):
        diff_file = [x for x in paths if "diff" in os.path.basename(x)][0]
        search_file = [x for x in paths if "sci" in os.path.basename(x)][0]
        tmpl_file = [x for x in paths if "temp" in os.path.basename(x)][0]

        # load and convert to tensor
        diff = np.load(os.path.join(diff_file), mmap_mode='r')
        srch = np.load(os.path.join(search_file), mmap_mode='r')
        tmpl = np.load(os.path.join(tmpl_file), mmap_mode='r')
        
        def StandardScalertorch(img):
            img = np.nan_to_num(img, nan=np.nanmean(img))
            img = RobustScaler().fit_transform(np.squeeze(img))
            img = np.nan_to_num(img, nan=np.nanmean(img))
            return img
        diff = StandardScalertorch(diff)
        srch = StandardScalertorch(srch)
        tmpl = StandardScalertorch(tmpl)
        
        helper = tmpl_file.split("/")[-1].split("_")
        ids = [int(helper[0])]
        ids = torch.tensor(ids)
        csv = self.feature_file[self.feature_file["diaSourceId"].isin([ids[0].item()])]
        labels = csv["real"].values
        labels = torch.tensor(labels).type(torch.FloatTensor).squeeze()
        
        diff = torch.tensor(diff).type(torch.FloatTensor).unsqueeze(0)
        srch = torch.tensor(srch).type(torch.FloatTensor).unsqueeze(0)
        tmpl = torch.tensor(tmpl).type(torch.FloatTensor).unsqueeze(0)
        
        return diff, srch, tmpl, labels, ids