import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from config import Config
from ours_model import (MultiHeadAttention, 
                        MultiHeadSelfAttention,
                        CADEmbedding,
                        TransformerEncoderLayer,
                        TransformerDecoderLayer,
                        TransformerDecoderLayerGlobalImproved,
                        TransformerEncoder,
                        TransformerDecoder,
                        ConstEmbedding,
                        CADFCN)

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()


class CMCAD(nn.Module):
    def __init__(self, config):
        super(CMCAD, self).__init__()
        self._config = config
        self._device = config.device

        self.encoder_embedding = CADEmbedding(config=config, need_pos=True)
        self.encoder = TransformerEncoder(config=config, transformer_layer=TransformerEncoderLayer)
        self.decoder_embedding = CADEmbedding(config=config, need_pos=True)
        self.const_embedding = ConstEmbedding(config)
        # self.decoder = TransformerDecoder(config=config, transformer_layer=TransformerDecoderLayer)
        self.decoder = TransformerDecoder(config=config, transformer_layer=TransformerDecoderLayerGlobalImproved)

        self.fcn = CADFCN(config=config)

        self.is_train = True

    def train_mode(self):
        self.train()
        self.is_train = True

    def predict_mode(self):
        self.eval()
        self.is_train = False

    def forward(self, enc_command, enc_args, enc_mask, dec_command=None, dec_args=None, dec_mask=None, z=None, mode='AE'):
        if mode is not 'DE':    # 如果decoder模式则不需要encoder
            enc_emb = self.encoder_embedding(enc_command, enc_args)
            enc_attn_mask = _make_attn_mask(mask=enc_mask).to(self._device)
            memory = self.encoder(inputs=enc_emb, mask=enc_attn_mask)
            z = torch.sum(memory * enc_mask.unsqueeze(2), dim=1, keepdim=True) / torch.sum(enc_mask, dim=1, keepdim=True).unsqueeze(2)

        if mode is 'EN':        # encoder模式，仅使用encoder
            return z
        if mode == 'AE':
            _dec_command = dec_command[:, :-1]
            _dec_args = dec_args[:, :-1]
            _dec_mask = dec_mask[:, :-1]

            dec_self_attn_mask = _make_decode_attn_mask(_dec_mask).to(self._device)
        # dec_emb = self.decoder_embedding(_dec_command, _dec_args)
        # dec_path_rep = self.decoder(inputs=dec_emb, mask=dec_self_attn_mask, memory=memory[:, :-1])
        dec_emb = self.const_embedding(z)[:, :-1]
        dec_path_rep = self.decoder(inputs=dec_emb, mask=None, memory=z)  # 传入隐向量z和decoder交互

        command_logits, args_logits = self.fcn(dec_path_rep)


        return command_logits, args_logits

