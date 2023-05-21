import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from config import Config


# def _get_visibility_mask(commands, seq_dim=0):
#     """
#     Args:
#         commands: Shape [S, ...]
#     """
#     S = commands.size(seq_dim)
#     with torch.no_grad():
#         visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) < S - 1
#
#         if seq_dim == 0:
#             return visibility_mask.unsqueeze(-1)
#         return visibility_mask



class CADLoss(nn.Module):
    def __init__(self, config:Config):
        super(CADLoss, self).__init__()

        self._config = config
        self.arg_dim = config.arg_dim + 1

        self.command_loss_fn = nn.CrossEntropyLoss() 
        self.args_loss_fn = nn.CrossEntropyLoss()
        self.register_buffer("cmd_args_mask", config.CMD_ARGS_MASK.to(config.device))

    def forward(self, command_logits, args_logits, command_tgt, args_tgt):
        """
        command_logits: batch_size * max_len - 1 * command_type_num
        args_logits: batch_size * max_len - 1 * args_num
        command_tgt: batch_size * max_len - 1
        args_tgt: batch_size * max_len - 1 * args_num
        """
        command_mask = (command_tgt == 4).cumsum(dim=-1) == 0
        args_mask = self.cmd_args_mask[command_tgt.long()]
        # args_mask = self._config.CMD_ARGS_MASK[(command_tgt).long()].to(args_logits)
        # args_logits *= args_mask.float()
        # if self._config.mode == "output_decoder":
        #     return args_logits

        command_loss = self.command_loss_fn(command_logits[command_mask.bool()].reshape(-1, self._config.command_type_num), (command_tgt[command_mask.bool()]).reshape(-1).long())  
        # args_loss = self.args_loss_fn(args_logits, args_tgt)
        args_loss = self.args_loss_fn(args_logits[args_mask.bool()].reshape(-1, self.arg_dim), args_tgt[args_mask.bool()].reshape(-1).long() + 1)

        return command_loss, args_loss
