import os
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
import time
import torchvision
from torchvision import datasets, models, transforms
import math, sys
from config import Config

from ours_dataset import CADDataset, ClassificationDataset
from ours_model import DeepCAD, CMCAD, CMCAD_Ablation
from ours_loss import CADLoss
from config import Config
from evaluate_acc import evaluate
from generate_dxf import generate_one


def dataset_split(config:Config):
    dataset = ClassificationDataset(config)
    length = len(dataset)
    indices = list(range(length))
    split = int(length * 0.1)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=dataset, batch_size=1, sampler=test_sampler)
    return train_loader, test_loader

def classification(config:Config):
    dataset = ClassificationDataset(config)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = DeepCAD(config=config)
    model.load_state_dict(torch.load(config.saved_model_path))
    model.predict_mode()
    model.to(config.device)

    clf_head = nn.Sequential(
        nn.Linear(256, 128),
        nn.Linear(128, 9)
    )
    clf_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(clf_head.parameters(), lr=1e-3)

    for epoch in range(50):
        clf_head.train()
        epoch_loss = []
        data_iter = tqdm(train_loader, desc="Iteration")
        for data in data_iter:
            optimizer.zero_grad()
            enc_command, enc_args, label, enc_mask = data

            with torch.no_grad():
                z = model(enc_command, enc_args, enc_mask, mode='EN')

            logits = clf_head(z).squeeze(1)
            y = F.one_hot(label, num_classes=9).float()
            assert logits.shape == y.shape
            loss = clf_loss(logits, y)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            sttr = 'epoch: {0}, loss: {1}'.format(epoch+1, round(np.mean(epoch_loss), 4))
            data_iter.set_description(sttr)

    data_iter = tqdm(test_loader, desc="Evaluation")
    acc = []
    for data in data_iter:
        with torch.no_grad():
            enc_command, enc_args, label, enc_mask = data
            z = model(enc_command, enc_args, enc_mask, mode='EN')
            logits = clf_head(z).squeeze(1)
            y_hat = torch.argmax(logits)
            acc.extend(y_hat == label)
    print(np.array(acc).mean())


    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    torch.save(clf_head.state_dict(), config.save_path + f'ClfCAD_{curr_time}.pkl')

if __name__ == '__main__':
    config = Config()
    classification(config)


