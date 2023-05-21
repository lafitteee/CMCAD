import numpy as np
from tqdm import tqdm
from config import Config

TOLERANCE = 5



def evaluate(cfg:Config, out_cmd, out_args, gt_cmd, gt_args):
    avg_cmd_acc = [] # ACC_CMD
    avg_args_acc = []   # ACC_ARGS
    valid_seq = []

    each_cmd_cnt = []
    each_cmd_acc = []

    args_mask = cfg.CMD_ARGS_MASK.numpy()
    out_cmd[out_cmd == 0] = 4
    for i in range(out_cmd.shape[0]):
        if out_cmd[i,-1] != 4:
            valid_seq.append(0)
            continue
        valid_seq.append(1)
        cmd_acc = (out_cmd[i] == gt_cmd[i]).astype(np.int32)
        args_acc = []
        for j in range(gt_cmd.shape[1]):
            cmd = gt_cmd[i, j]
            if cmd in [0, 4]:
                continue

            if out_cmd[i][j] == gt_cmd[i][j]:
                tole_acc = (np.abs(out_args[i][j] - gt_args[i][j]) < TOLERANCE).astype(np.int32)
                
                valid_args_acc = tole_acc[args_mask[cmd].astype(np.bool_)].tolist()
                args_acc.extend(valid_args_acc)
    
        args_acc = np.nanmean(args_acc)
        avg_args_acc.append(args_acc)
        cmd_acc = np.nanmean(cmd_acc)
        avg_cmd_acc.append(cmd_acc)
    
    avg_cmd_acc = np.nanmean(avg_cmd_acc)
    avg_args_acc = np.nanmean(avg_args_acc)
    valid_seqs_rate = np.nanmean(valid_seq)
    # print("avg cmds acc (ACC_cmd):", avg_cmd_acc)
    # print("avg args acc (ARGS_cmd):", avg_args_acc)
    return avg_cmd_acc, avg_args_acc, valid_seqs_rate
