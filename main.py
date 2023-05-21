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
import sys

from ours_dataset import CADDataset, SimilarityDataset
from ours_model import DeepCAD, CMCAD, CMCAD_Ablation
from ours_loss import CADLoss
from config import Config
from evaluate_acc import evaluate
from generate_dxf import generate_one


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def dataset_split(config:Config):
    dataset = CADDataset(config)
    length = len(dataset)
    indices = list(range(length))
    split = int(length * 0.1)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=dataset, batch_size=1, sampler=test_sampler)
    return train_loader, test_loader

def main(config):
    loss_fn = CADLoss(config=config)
    model = DeepCAD(config=config).to(config.device)

    train_dataloader, test_dataloader = dataset_split(config)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)

    train_loss_arr = []
    test_loss_arr = []
    acc_arr = []

    for epoch in range(config.epoch):
        model.train_mode()

        epoch_loss = []
        epoch_cmd_loss = []
        epoch_arg_loss = []
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for data in epoch_iterator:
            optimizer.zero_grad()

            enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, _, _ = data
            command_logits, args_logits = model(enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask)
            command_loss, args_loss = loss_fn(command_logits, args_logits, dec_command[:, 1:], dec_args[:, 1:])

            (command_loss + args_loss).backward()
            optimizer.step()
            batch_loss = (command_loss + args_loss).item()
            train_loss_arr.append(round(batch_loss, 4))
            epoch_cmd_loss.append(command_loss.item())
            epoch_arg_loss.append(args_loss.item())
            epoch_loss.append((command_loss + args_loss).item())
            sttr = 'epoch: {0}, command_loss: {1}, args_loss: {2}, loss: {3}'.format(epoch+1, round(np.nanmean(epoch_cmd_loss), 4), round(np.nanmean(epoch_arg_loss), 4), round(np.nanmean(epoch_loss), 4))
            epoch_iterator.set_description(sttr)

        scheduler.step()

        # Evaluation
        if (epoch + 1) % 5 == 0:
            cmds_acc = []
            args_acc = []
            with torch.no_grad():
                model.predict_mode()
                test_loss = []
                epoch_iterator = tqdm(test_dataloader, desc="Evaluation")
                for data in epoch_iterator:
                    enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img, _ = data
                    command_logits, args_logits = model(enc_command, enc_args, enc_mask)
                    command_loss, args_loss = loss_fn(command_logits, args_logits, dec_command[:, 1:], dec_args[:, 1:])
                    batch_loss = (command_loss.cpu() + args_loss.cpu()).item()
                    test_loss.append(batch_loss)

                    cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
                    arg_mask = config.CMD_ARGS_MASK.to(config.device)[enc_command[:, 1:].long()]
                    cmd = torch.argmax(command_logits, dim=2).cpu() * cmd_mask.cpu()
                    args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()

                    batch_cmd_acc, batch_args_acc, _ = evaluate(config, cmd.numpy(), args.numpy(),
                                                             enc_command[:, 1:].cpu().numpy(),
                                                             enc_args[:, 1:].cpu().numpy())

                    cmds_acc.append(batch_cmd_acc)
                    args_acc.append(batch_args_acc)

            test_loss_arr.append(round(np.nanmean(test_loss), 4))
            acc_arr.append(round(np.nanmean(args_acc), 4))

            print("avg cmds acc (ACC_cmd):", np.mean(cmds_acc))
            print("avg args acc (ARGS_cmd):", np.mean(args_acc))
            print("TEST LOSS:", np.nanmean(test_loss))
            
        if (epoch + 1) % config.save_per_epoch == 0:
            curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            torch.save(model.state_dict(), config.save_path + f'DeepSVG_{curr_time}.pkl')
    statics = {
        'train_loss': train_loss_arr,
        'test_loss': test_loss_arr,
        'acc': acc_arr,
    }
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    np.save(config.save_path + f'DeepSVG_training_statics_{curr_time}.npy', np.array(statics), allow_pickle=True)


def img_trainer(config:Config):

    seqs_model = DeepCAD(config)
    seqs_model.load_state_dict(torch.load(config.saved_model_path))
    seqs_model.to(config.device)

    imgs_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    fc_features = imgs_model.fc.in_features
    imgs_model.fc = nn.Linear(fc_features, config.d_model)
    imgs_model.to(config.device)
    
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    train_loader, test_loader = dataset_split(config)
    optimizer = optim.AdamW(imgs_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)

    train_loss_arr = []
    test_loss_arr = []
    acc_arr = []
    for epoch in range(config.epoch):
        seqs_model.predict_mode()
        imgs_model.train()

        epoch_loss = []
        epoch_iter = tqdm(train_loader)
        for data in epoch_iter:
            optimizer.zero_grad()
            enc_command, enc_args, enc_mask, _, _, _, img, _ = data
            with torch.no_grad():
                rep = seqs_model(enc_command, enc_args, enc_mask, mode='EN').squeeze_()
            
            output = imgs_model(img)
            img_loss = kl_loss(F.log_softmax(output, dim=1), F.softmax(rep, dim=1))
            img_loss.backward()
            optimizer.step()

            epoch_loss.append(img_loss.item())
            sttr = 'epoch: {0}, lr: {1} img_loss: {2}'.format(epoch + 1, optimizer.param_groups[0]['lr'],
                                                              round(np.mean(epoch_loss), 4))
            epoch_iter.set_description(sttr)

        scheduler.step()

        # 每5轮在测试集上做测试
        if (epoch + 1) % 5 == 0:
            cmds_acc = []
            args_acc = []
            test_iter = tqdm(test_loader)
            for test_data in test_iter:
                enc_command, enc_args, enc_mask, _, _, _, img, _ = test_data
                with torch.no_grad():
                    output = imgs_model(img)
                    cmds_logits, args_logits = seqs_model(None, None, None, z=output.unsqueeze(1), mode='DE')

                    cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
                    arg_mask = config.CMD_ARGS_MASK.to(config.device) [enc_command[:, 1:].long()]
                    cmd = torch.argmax(cmds_logits, dim=2).cpu() * cmd_mask.cpu()
                    args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()
                    batch_cmd_acc, batch_args_acc, _ = evaluate(config, cmd.numpy(), args.numpy(), enc_command[:, 1:].cpu().numpy(), enc_args[:, 1:].cpu().numpy())
                    cmds_acc.append(batch_cmd_acc)
                    args_acc.append(batch_args_acc)
            print("avg cmds acc (ACC_cmd):", np.nanmean(cmds_acc))
            print("avg args acc (ARGS_cmd):", np.nanmean(args_acc))
            acc_arr.append(round(np.nanmean(args_acc), 4))
    print("===============Saving Model===============")
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = f'学生模型_基于DeepSVG_{curr_time}_acc_{round(acc_arr[-1], 2)}.pkl'
    torch.save(imgs_model.state_dict(), config.save_path + file_name)

    statics = {
        'train_loss': train_loss_arr,
        # 'test_loss': test_loss_arr,
        'acc': acc_arr,
    }

    print("===============Saving Results===============")
    np.save(config.save_path + f'反演loss_and_acc_基于DeepSVG_{curr_time}.npy', np.array(statics), allow_pickle=True)


def eval_img_trainer(config: Config):
    seqs_model = DeepCAD(config)
    seqs_model.load_state_dict(torch.load(config.saved_model_path))
    seqs_model.to(config.device)

    imgs_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    fc_features = imgs_model.fc.in_features
    imgs_model.fc = nn.Linear(fc_features, config.d_model)
    imgs_model.load_state_dict(torch.load(config.img_encoder_model_path))
    imgs_model.to(config.device)

    dataset = CADDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    train_loss_arr = []
    test_loss_arr = []
    acc_arr = []
    seqs_model.predict_mode()
    imgs_model.eval()

    cmds_acc = []
    args_acc = []
    test_iter = tqdm(dataloader)
    for test_data in test_iter:
        enc_command, enc_args, enc_mask, _, _, _, img, _ = test_data
        with torch.no_grad():
            output = imgs_model(img)
            cmds_logits, args_logits = seqs_model(None, None, None, z=output.unsqueeze(1), mode='DE')

            cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
            arg_mask = config.CMD_ARGS_MASK.to(config.device)[enc_command[:, 1:].long()]
            cmd = torch.argmax(cmds_logits, dim=2).cpu() * cmd_mask.cpu()
            args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()
            batch_cmd_acc, batch_args_acc, _ = evaluate(config, cmd.numpy(), args.numpy(),
                                                        enc_command[:, 1:].cpu().numpy(),
                                                        enc_args[:, 1:].cpu().numpy())
            cmds_acc.append(batch_cmd_acc)
            args_acc.append(batch_args_acc)
    print("avg cmds acc (ACC_cmd):", np.nanmean(cmds_acc))
    print("avg args acc (ARGS_cmd):", np.nanmean(args_acc))
    acc_arr.append(round(np.nanmean(args_acc), 4))


def eval_main(config):
    dataset = CADDataset(config=config)
    # loss_fn = CADLoss(config=config)
    model = DeepCAD(config=config)
    model.load_state_dict(torch.load(config.saved_model_path))
    # torch.save(model.state_dict(), './save/saved_pretrain_epoch_49_resave.pkl', _use_new_zipfile_serialization=False)  # 训练所有数据后，保存网络的参数
    model.to(config.device)

    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)

    representations = []
    with torch.no_grad():
        model.predict_mode()

        epoch_iterator = tqdm(dataloader, desc="Generating Representations")
        for data in epoch_iterator:
            # optimizer.zero_grad()
            enc_command, enc_args, enc_mask, _, _, _ = data
            # print(enc_args)
            rep = model(enc_command, enc_args, enc_mask)
            
            # command_loss, args_loss = loss_fn(command_logits, args_logits, dec_command[:, 1:], dec_args[:, 1:])
            representations.append(rep)
        
        representations = torch.cat(representations, dim=0) # size:[17725, 1, 256]
        representations = torch.squeeze(representations, dim=1) #[17725, 256]

        dot_product = representations.matmul(representations.permute(1, 0))
        norm = torch.norm(representations, p=2, dim=-1, keepdim=True)
        norm_norm = norm.matmul(norm.permute(1, 0))
        cosine_similarity = dot_product / norm_norm

        _, topK = torch.topk(cosine_similarity, k=10, sorted=True, dim=-1)
        topK = topK.cpu().numpy()

        np.save(config.topK_path, topK)


def eval_main_decoder_output(config:Config):
    dataset = CADDataset(config)
    model = DeepCAD(config=config)
    model.load_state_dict(torch.load(config.saved_model_path))
    model.to(config.device)

    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)

    output_cmds = []
    output_args = []
    cmds_acc = []
    args_acc = []
    with torch.no_grad():
        model.predict_mode()

        epoch_iterator = tqdm(dataloader, desc="Generating Representations")
        for data in epoch_iterator:
            enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img = data
            # print(enc_args)
            command_logits, args_logits = model(enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask)
            
            cmd_mask = (dec_command[:, 1:] == 4).cumsum(dim=-1) == 0
            arg_mask = config.CMD_ARGS_MASK[dec_command[:, 1:].long()]
            cmd = torch.argmax(command_logits, dim=2).cpu() * cmd_mask.cpu()
            args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()
            output_cmds.extend(cmd.cpu().detach().numpy())
            output_args.extend(args.cpu().detach().numpy())

            batch_cmd_acc, batch_args_acc = evaluate(config, cmd.numpy(), args.numpy(), enc_command[:, 1:].cpu().numpy(), enc_args[:, 1:].cpu().numpy())
            # sttr = 'epoch: {0}, command_loss: {1}, args_loss: {2}, loss: {3}'.format(epoch, command_loss.item(), args_loss.item(), (command_loss + args_loss).item())
            # epoch_iterator.set_description(sttr)
            cmds_acc.append(batch_cmd_acc)
            args_acc.append(batch_args_acc)

    print("avg cmds acc (ACC_cmd):", np.mean(cmds_acc))
    print("avg args acc (ARGS_cmd):", np.mean(args_acc))
    # np.save(config.decoder_output_path, np.array(output_cmds), allow_pickle=True)
    # np.save(config.decoder_output_path, np.array(output_args), allow_pickle=True)

def show_tensor(t):
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    a = t.numpy()
    print(a)



def train_cross_modal_ae(config:Config):
    loss_func = CADLoss(config)
    
    model = CMCAD_Ablation(config).to(config.device)

    # dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    train_dataloader, test_dataloader = dataset_split(config)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.85)

    best_model = None
    best_loss = None
    train_loss_arr = []
    test_loss_arr = []
    acc_arr = []

    for epoch in range(config.epoch):
        model.train_mode()
        epoch_cmd_loss = []
        epoch_arg_loss = []
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        
        for data in epoch_iterator:
            optimizer.zero_grad()

            enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img = data
            command_logits, args_logits = model(enc_command, enc_args, enc_mask, img, dec_command, dec_args, dec_mask)
            command_loss, args_loss = loss_func(command_logits, args_logits, dec_command[:, 1:], dec_args[:, 1:])

            (command_loss + args_loss).backward()
            optimizer.step()

            batch_loss = (command_loss.cpu() + args_loss.cpu()).item()
            train_loss_arr.append(round(batch_loss, 4))

            epoch_cmd_loss.append(command_loss.item())
            epoch_arg_loss.append(args_loss.item())
            sttr = 'epoch: {0}, command_loss: {1}, args_loss: {2}, loss: {3}'.format(epoch+1, round(np.mean(epoch_cmd_loss), 4), round(np.mean(epoch_arg_loss), 4), round(np.mean(epoch_cmd_loss)+np.mean(epoch_arg_loss), 4))
            epoch_iterator.set_description(sttr)

        # if best_loss is None or epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     best_model = model.state_dict()
        
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            cmds_acc = []
            args_acc = []
            with torch.no_grad():
                model.predict_mode()
                test_loss = []
                epoch_iterator = tqdm(test_dataloader, desc="Evaluation")
                for data in epoch_iterator:
                    enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img = data
                    # print(enc_args)
                    command_logits, args_logits = model(enc_command, enc_args, enc_mask, img)
                    command_loss, args_loss = loss_func(command_logits, args_logits, dec_command[:, 1:], dec_args[:, 1:])
                    batch_loss = (command_loss.cpu() + args_loss.cpu()).item()
                    test_loss.append(batch_loss)

                    cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
                    arg_mask = config.CMD_ARGS_MASK.to(config.device)[enc_command[:, 1:].long()]
                    cmd = torch.argmax(command_logits, dim=2).cpu() * cmd_mask.cpu()
                    args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()

                    batch_cmd_acc, batch_args_acc, _ = evaluate(config, cmd.numpy(), args.numpy(),
                                                             enc_command[:, 1:].cpu().numpy(),
                                                             enc_args[:, 1:].cpu().numpy())

                    cmds_acc.append(batch_cmd_acc)
                    args_acc.append(batch_args_acc)

            test_loss_arr.append(round(np.nanmean(test_loss), 4))
            acc_arr.append(round(np.nanmean(args_acc), 4))

            print("avg cmds acc (ACC_cmd):", np.mean(cmds_acc))
            print("avg args acc (ARGS_cmd):", np.mean(args_acc))
            print("TEST LOSS:", np.nanmean(test_loss))


        # if (epoch + 1) % config.save_per_epoch == 0:
        #     curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        #     file_name = f'教师模型_拼接融合_{curr_time}_acc_{round(acc_arr[-1], 2)}.pkl'
        #     torch.save(model.state_dict(), config.save_path + file_name)
    print("===============Saving Model===============")
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = f'教师模型_拼接融合_{curr_time}_acc_{round(acc_arr[-1], 2)}.pkl'
    torch.save(model.state_dict(), config.save_path + file_name)

    print("===============Saving Results==============")
    statics = {
        'train_loss': train_loss_arr,
        'test_loss': test_loss_arr,
        'acc': acc_arr,
    }
    # curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    np.save(config.save_path + f'学生模型训练过程acc_基于拼接融合{curr_time}.npy', np.array(statics), allow_pickle=True)




def eval_cross_modal_ae(config:Config):
    model = CMCAD_Ablation(config).to(config.device)
    model.load_state_dict(torch.load(config.CMCAD_model_path))
    # train_dataloader, test_dataloader = dataset_split(config)
    dataset = CADDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    output_cmds = []
    output_args = []
    cmds_acc = []
    args_acc = []
    with torch.no_grad():
        model.predict_mode()

        epoch_iterator = tqdm(dataloader, desc="Generating Representations")
        for data in epoch_iterator:
            enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img = data
            # print(enc_args)
            command_logits, args_logits = model(enc_command, enc_args, enc_mask, img)
            
            cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
            arg_mask = config.CMD_ARGS_MASK[enc_command[:, 1:].long()]
            cmd = torch.argmax(command_logits, dim=2).cpu() * cmd_mask.cpu()
            args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()
            output_cmds.extend(cmd.cpu().detach().numpy())
            output_args.extend(args.cpu().detach().numpy())

            batch_cmd_acc, batch_args_acc = evaluate(config, cmd.numpy(), args.numpy(), enc_command[:, 1:].cpu().numpy(), enc_args[:, 1:].cpu().numpy())

            cmds_acc.append(batch_cmd_acc)
            args_acc.append(batch_args_acc)

    print("avg cmds acc (ACC_cmd):", np.mean(cmds_acc))
    print("avg args acc (ARGS_cmd):", np.mean(args_acc))

    print("===============Saving Results===============")
    curr_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    np.save(config.decoder_output_path + f'output_cmds_{curr_time}.npy', np.array(output_cmds), allow_pickle=True)
    np.save(config.decoder_output_path + f'output_args_{curr_time}.npy', np.array(output_args), allow_pickle=True)
    print("==================Complete==================")


def train_img_encoder_base_CMCAD(config:Config):
    """
        CMCAD AE模型作为基础模型,训练图像编码器将图像编码逼近AE的隐空间;
        测试图像转化为序列的准确率
    """
    print("training STUDENT model based multi-modal teacher")
    train_dataloader, test_dataloader = dataset_split(config)

    """消融实验，模态特征直接相加"""
    model = CMCAD_Ablation(config).to(config.device)
    model.load_state_dict(torch.load(config.teacher_model_path))
    """基础实验，交叉注意力特征融合"""
    # model = CMCAD(config).to(config.device)
    # model.load_state_dict((torch.load(config.teacher_model_path)))

    imgs_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    fc_features = imgs_model.fc.in_features
    imgs_model.fc = nn.Linear(fc_features, config.d_model)
    imgs_model.to(config.device)
    
    # img_loss_func = nn.MSELoss()
    """使用KL散度作为损失函数"""
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(imgs_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)

    train_loss_arr = []
    test_loss_arr = []
    acc_arr = []

    for epoch in range(config.epoch):
        model.predict_mode()
        imgs_model.train()
        epoch_loss = []
        epoch_iter = tqdm(train_dataloader)
        for data in epoch_iter:
            optimizer.zero_grad()
            enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img = data
            with torch.no_grad():
                rep = model(enc_command, enc_args, enc_mask, img, mode='EN').squeeze_()
            
            output = imgs_model(img)
            img_loss = kl_loss(F.log_softmax(output, dim=1), F.softmax(rep, dim=1))
            img_loss.backward()
            optimizer.step()
                
            epoch_loss.append(img_loss.item())
            sttr = 'epoch: {0}, lr: {1} img_loss: {2}'.format(epoch+1, optimizer.param_groups[0]['lr'], round(np.mean(epoch_loss), 4))
            epoch_iter.set_description(sttr)

        scheduler.step()
        # 每5轮在测试集上做测试
        if (epoch + 1) % 5 == 0:
            cmds_acc = []
            args_acc = []
            valid_seqs_rate = []
            test_iter = tqdm(test_dataloader)
            for test_data in test_iter:
                enc_command, enc_args, enc_mask, _, _, _, img = test_data
                with torch.no_grad():
                    output = imgs_model(img)
                    cmds_logits, args_logits = model(None, None, None, None, z=output.unsqueeze(1), mode='DE')

                    cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
                    arg_mask = config.CMD_ARGS_MASK.to(config.device)[enc_command[:, 1:].long()]
                    cmd = torch.argmax(cmds_logits, dim=2).cpu() * cmd_mask.cpu()
                    args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()
                    batch_cmd_acc, batch_args_acc, batch_valid_seqs_rate = evaluate(config, cmd.numpy(), args.numpy(), enc_command[:, 1:].cpu().numpy(), enc_args[:, 1:].cpu().numpy())
                    cmds_acc.append(batch_cmd_acc)
                    args_acc.append(batch_args_acc)
                    valid_seqs_rate.append(batch_valid_seqs_rate)

            # test_loss_arr.append(round(np.nanmean(test_loss), 4))
            acc_arr.append(round(np.nanmean(args_acc), 4))

            print("valid seq rate:", np.nanmean(valid_seqs_rate))
            print("avg cmds acc (ACC_cmds):", np.nanmean(cmds_acc))
            print("avg args acc (ACC_args):", np.nanmean(args_acc))

        # if (epoch + 1) % config.save_per_epoch == 0:
        #     curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        #     file_name = f'学生模型_基于加法融合_{curr_time}_acc_{round(acc_arr[-1], 2)}.pkl'
        #     torch.save(imgs_model.state_dict(), config.save_path + file_name)

    print("===============Saving Model===============")
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = f'学生模型_基于拼接融合_{curr_time}_acc_{round(acc_arr[-1], 2)}.pkl'
    torch.save(imgs_model.state_dict(), config.save_path + file_name)

    statics = {
        'train_loss': train_loss_arr,
        # 'test_loss': test_loss_arr,
        'acc': acc_arr,
    }

    print("===============Saving Results===============")
    np.save(config.save_path + f'反演loss_and_acc_基于拼接融合_{curr_time}.npy', np.array(statics), allow_pickle=True)



def eval_img_encoder_base_CMCAD(config:Config):
    """
        输入图像,输出dxf序列,并计算命令和参数准确率,把序列保存
        编码器为:仅图像编码器
        解码器为:CMCAD的解码器
    """
    model = CMCAD(config)
    # model = CMCAD_Ablation(config)
    model.load_state_dict(torch.load(config.teacher_model_path))
    model.to(config.device)

    img_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    fc_features = img_model.fc.in_features
    img_model.fc = nn.Linear(fc_features, config.d_model)
    img_model.load_state_dict(torch.load(config.img_encoder_model_path))
    img_model.to(config.device)

    _, test_loader = dataset_split(config)
    output_cmds = []
    output_args = []

    train_loss_arr = []
    test_loss_arr = []
    acc_arr = []
    cmds_acc = []
    args_acc = []
    valid_seqs_rate = []

    # 测试集做推理，保存每个样本的id和它的参数准确率
    id_and_acc = []
    with torch.no_grad():
        img_model.eval()

        epoch_iterator = tqdm(test_loader, desc="Generating Representations")
        for data in epoch_iterator:
            # enc_command, enc_args, enc_mask, dec_command, dec_args, dec_mask, img = data
            # rep = img_model(img).unsqueeze(1)
            # command_logits, args_logits = model(None, None, None, None, z=rep, mode='DE')
            #
            # cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
            # arg_mask = config.CMD_ARGS_MASK[enc_command[:, 1:].long()]
            # cmd = torch.argmax(command_logits, dim=2).cpu() * cmd_mask.cpu()
            # args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()
            # output_cmds.extend(cmd.cpu().detach().numpy())
            # output_args.extend(args.cpu().detach().numpy())
            #
            # batch_cmd_acc, batch_args_acc = evaluate(config, cmd.numpy(), args.numpy(), enc_command[:, 1:].cpu().numpy(), enc_args[:, 1:].cpu().numpy())
            #
            # cmds_acc.append(batch_cmd_acc)
            # args_acc.append(batch_args_acc)
            enc_command, enc_args, enc_mask, _, _, _, img, idx = data
            output = img_model(img)

            cmds_logits, args_logits = model(None, None, None, None, z=output.unsqueeze(1), mode='DE')

            cmd_mask = (enc_command[:, 1:] == 4).cumsum(dim=-1) == 0
            arg_mask = config.CMD_ARGS_MASK.to(config.device)[enc_command[:, 1:].long()]
            cmd = torch.argmax(cmds_logits, dim=2).cpu() * cmd_mask.cpu()
            args = (torch.argmax(args_logits, dim=3) - 1).cpu() * arg_mask.cpu()

            """生成DXF文件"""
            generate_one(cmd[0].cpu().numpy(), args[0].cpu().numpy(), idx[0].item(), save_dir='img2dxf_CROSS_test')

            # output_cmds.extend(cmd.cpu().detach().numpy())
            # output_args.extend(args.cpu().detach().numpy())
            output_cmds.extend(enc_command.cpu().detach().numpy())
            output_args.extend(enc_args.cpu().detach().numpy())

            batch_cmd_acc, batch_args_acc, batch_valid_seqs_rate = evaluate(config, cmd.numpy(), args.numpy(),
                                                                            enc_command[:, 1:].cpu().numpy(),
                                                                            enc_args[:, 1:].cpu().numpy())
            cmds_acc.append(batch_cmd_acc)
            args_acc.append(batch_args_acc)
            valid_seqs_rate.append(batch_valid_seqs_rate)

            id_and_acc.append({'id': idx[0].item(), 'acc': batch_args_acc})

    acc_arr.append(round(np.nanmean(args_acc), 4))

    print("valid seq rate:", np.nanmean(valid_seqs_rate))
    print("avg cmds acc (ACC_cmds):", np.nanmean(cmds_acc))
    print("avg args acc (ACC_args):", np.nanmean(args_acc))

    statics = {
        'train_loss': train_loss_arr,
        # 'test_loss': test_loss_arr,
        'acc': acc_arr,
    }
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # np.save(config.save_path + f'反演_全数据集_loss_and_acc_拼接融合_t=5_{curr_time}.npy', np.array(statics), allow_pickle=True)
    np.save('./generated_dxf/img2dxf_CROSS_test/id_and_acc.npy', np.array(id_and_acc), allow_pickle=True)
    # print("===============Saving Results===============")
    # curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # np.save(config.decoder_output_path + f'img2dxf_cmds_GroundTruth_{curr_time}.npy', np.array(output_cmds), allow_pickle=True)
    # np.save(config.decoder_output_path + f'img2dxf_args_GroundTruth_{curr_time}.npy', np.array(output_args), allow_pickle=True)
    # print("==================Complete==================")


def similarity(config:Config):
    dataset = SimilarityDataset(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = DeepCAD(config)
    model.load_state_dict(torch.load(config.saved_model_path))
    model.to(config.device)

    representations = []
    with torch.no_grad():
        dataiter = tqdm(dataloader, desc="Generating Representations")
        for data in dataiter:
            enc_command, enc_args, enc_mask = data

            z = model(enc_command, enc_args, enc_mask, mode='EN').squeeze_().detach().cpu().numpy()
            if z.shape[0] != 256:
                print(z.shape)
            representations.append(z)

    np.save('./representations.npy', np.array(representations), allow_pickle=True)


if __name__ == '__main__':
    
    setup_seed(42)
    cfg = Config()
    if cfg.mode == "train":
        # print("===============Training Teacher Model===============")
        # main(cfg)
        print("===============Training Student Model===============")
        img_trainer(cfg)
    elif cfg.mode == "image_trainer":
        img_trainer(cfg)
    elif cfg.mode == "eval":
        eval_img_trainer(cfg)
    elif cfg.mode == "output_decoder":
        eval_main_decoder_output(cfg)
    elif cfg.mode == 'train_CMCAD':
        train_cross_modal_ae(cfg)
    elif cfg.mode == 'eval_CMCAD':
        eval_cross_modal_ae(cfg)
    elif cfg.mode == 'train_img_encoder_base_CMCAD':
        train_img_encoder_base_CMCAD(cfg)
    elif cfg.mode == 'eval_img_encoder_base_CMCAD':
        eval_img_encoder_base_CMCAD(cfg)
    elif cfg.mode == 'calculate_similarity':
        similarity(cfg)
