import numpy as np
import torch
from config import Config
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL
import PIL.Image as Image
from tqdm import tqdm
import glob


class CADDataset(Dataset):
    def __init__(self, config:Config):
        super(CADDataset, self).__init__()

        all_data = np.load(config.all_data_path, allow_pickle=True)
        cmds_np = all_data['cmds']
        args_np = all_data['args'].astype(np.int32)
        imgs_np = all_data['imgs']
        mask_np = np.where(cmds_np == 4, 0, 1)
        # a = np.ones((mask_np.shape[0], 1), dtype=np.int32)
        # b = mask_np[:, :-1]
        # mask_np = np.concatenate((a, b), axis=1)

        self.cmds = torch.from_numpy(cmds_np).long()  # len * max_path
        self.args = torch.from_numpy(args_np).long()  # len * max_path * arg_num
        self.imgs = torch.from_numpy(imgs_np)
        self.mask = torch.from_numpy(mask_np).int()

        self.len = cmds_np.shape[0]
        self.device = config.device

        # id与文件名id的映射关系
        self.idx_map = np.load('indexs.npy', allow_pickle=True)

    def __getitem__(self, idx):
        return self.cmds[idx].to(self.device), self.args[idx].to(self.device), self.mask[idx].to(self.device),\
            self.cmds[idx].to(self.device), self.args[idx].to(self.device), self.mask[idx].to(self.device), self.imgs[idx].to(self.device), self.idx_map[idx]
    
    def __len__(self):
        return self.len


class ClassificationDataset(Dataset):
    def __init__(self, config:Config):
        super(ClassificationDataset, self).__init__()
        data = np.load(config.clf_data_dir, allow_pickle=True)
        cmds = np.int32(data['cmd'])
        args = np.int32(data['param'])
        labels = np.int32(data['label'])
        mask_np = np.where(cmds == 4, 0, 1)
        self.cmds = torch.from_numpy(cmds).long()  # len * max_path
        self.args = torch.from_numpy(args).long()  # len * max_path * arg_num
        self.labels = torch.from_numpy(labels).long()
        self.mask = torch.from_numpy(mask_np).int()
        self.len = cmds.shape[0]
        self.device = config.device

    def __getitem__(self, idx):
        return self.cmds[idx].to(self.device), self.args[idx].to(self.device), self.labels[idx].to(self.device), self.mask[idx].to(self.device)

    def __len__(self):
        return self.len


class SimilarityDataset(Dataset):
    def __init__(self, config:Config):
        super(SimilarityDataset).__init__()
        data = np.load(config.sim_data_dir, allow_pickle=True)
        cmds = np.int32(data['cmds'])
        args = np.int32(data['args'])
        mask_np = np.where(cmds == 4, 0, 1)
        self.cmds = torch.from_numpy(cmds).long()  # len * max_path
        self.args = torch.from_numpy(args).long()  # len * max_path * arg_num
        self.mask = torch.from_numpy(mask_np).int()
        self.len = cmds.shape[0]
        self.device = config.device

    def __getitem__(self, idx):
        return self.cmds[idx].to(self.device), self.args[idx].to(self.device), self.mask[idx].to(self.device)

    def __len__(self):
        return self.len


class CADDataset_bak(Dataset):
    def __init__(self, config):
        self._config = config

        command_np = np.load(config.command_file_path, allow_pickle=True)
        print('command_np.shape: ', command_np.shape)
        path_np = np.load(config.path_file_path, allow_pickle=True).astype(np.int32)
        print('path_np.shape: ', path_np.shape)
        # images = 
        # path_np = np.where(path_np == None, 0.0, path_np).astype(np.float32)
        # mask_np = np.where(command_np == -1, 0.0, 1.0)
        mask_np = np.where(command_np == 4, 0, 1)

        self.command = torch.from_numpy(command_np).long()  # len * max_path
        self.path = torch.from_numpy(path_np).long()  # len * max_path * arg_num
        self.mask = torch.from_numpy(mask_np).int()

        self.len = command_np.shape[0]

    def __getitem__(self, index):
        return self.command[index].to(self._config.device), self.path[index].to(self._config.device), self.mask[index].to(self._config.device),\
               self.command[index].to(self._config.device), self.path[index].to(self._config.device), self.mask[index].to(self._config.device)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    cfg = Config()
    # images = load_images(cfg)
    # print(len(images))