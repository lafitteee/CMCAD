import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import math, sys
from config import Config


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _make_attn_mask(mask):
    batch_size, max_len = mask.size()
    return mask.unsqueeze(1).expand(batch_size, max_len, max_len)


def _make_decode_attn_mask(mask):
    batch_size, max_len = mask.size()
    mask1 = _make_attn_mask(mask=mask)
    mask2 = torch.triu(torch.ones(max_len, max_len)).transpose(0, 1).unsqueeze(0).expand(batch_size, max_len, max_len).to(mask1)

    return torch.where((mask1 + mask2) > 1, 1.0, 0.0)


class PositionalEmbedding(nn.Module):
    def __init__(self, config:Config):
        super(PositionalEmbedding, self).__init__()
        self.position = torch.stack([torch.arange(0, config.max_path, dtype=torch.long) for _ in range(config.batch_size)], dim=0).to(config.device)
        self.embedding = nn.Embedding(config.max_path, config.d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_in")

    def forward(self):
        return self.embedding(self.position)


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg:Config):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = cfg.max_path

        self.PE = PositionalEmbedding(cfg)

    def forward(self, z):
        N = z.size(0)
        src = z.new_zeros(N, self.seq_len, self.d_model)
        pe = self.PE()[:src.shape[0], :src.shape[1], :]
        return src + pe


class CADEmbedding(nn.Module):
    def __init__(self, config:Config, need_pos=False):
        super(CADEmbedding, self).__init__()
        self.command_embedding = nn.Embedding(config.command_type_num, config.d_model)
        
        args_dim = config.arg_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * config.arg_num, config.d_model)
        # 控制是否需要 PositionalEmbedding
        self._need_pos = need_pos

        if need_pos:
            self.position_embedding = PositionalEmbedding(config=config)

    def forward(self, commands, args):
        N, S, _ = args.shape
        if self._need_pos:
            return self.command_embedding(commands) + \
                    self.embed_fcn(self.arg_embed((args + 1).long()).view(N, S, -1)) + \
                    self.position_embedding()[:commands.shape[0], :commands.shape[1], :]
        else:
            return self.command_embedding(commands) + self.embed_fcn(self.arg_embed((args + 1).long()).view(N, S, -1))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadSelfAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

        self.W_o = torch.nn.Linear(d_model, d_model)

        self._h = h
        self.head_size = d_model // h

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)  # [batch_size * num_head, length, q]
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)  # [batch_size * num_head, length, q]
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)  # [batch_size * num_head, length, v]

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self._h)], dim=0)

            score += (1 - mask) * -1000000000.0

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)  # [batch_size, length, v * num_head]

        self_attention = self.W_o(attention_heads)
        # self_attention = attention_heads

        return self_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

        self.W_o = torch.nn.Linear(d_model, d_model)

        self._h = h
        self.head_size = d_model // h

        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, y, mask):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)  # [batch_size * num_head, length, q]
        K = torch.cat(self.W_k(y).chunk(self._h, dim=-1), dim=0)  # [batch_size * num_head, length, q]
        V = torch.cat(self.W_v(y).chunk(self._h, dim=-1), dim=0)  # [batch_size * num_head, length, v]

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        # self.score = score
        mask = torch.cat([mask for _ in range(self._h)], dim=0)

        score += (1 - mask) * -1000000000.0

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)  # [batch_size, length, v * num_head]

        self_attention = self.W_o(attention_heads)

        return self_attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu", self_attn=None):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout) if self_attn is None else self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, mask):
        src2 = self.self_attn(src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)
        self.multi_head_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.linear_global = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, mask):
        tgt2 = self.self_attn(tgt, mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multi_head_attn(tgt, memory, mask)
        tgt = tgt + self.dropout2(tgt2)  # implicit broadcast
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderLayerGlobalImproved(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayerGlobalImproved, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)

        self.linear_global = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerGlobalImproved, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, *args, **kwargs):
        tgt1 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt1, mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.linear_global(memory)   # [61, 1 256]
        tgt = tgt + self.dropout2(tgt2)  # implicit broadcast

        tgt1 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt1))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, config, transformer_layer, n_layer=None, self_attn=None):
        super(TransformerEncoder, self).__init__()
        self.n_layer = n_layer if n_layer is not None else config.n_layer
        self.layers = nn.ModuleList([transformer_layer(config.d_model, config.nhead, self_attn=self_attn) for _ in range(self.n_layer)])

    def forward(self, inputs, mask):
        outputs = inputs

        for mod in self.layers:
            outputs = mod(src=outputs, mask=mask)

        return outputs


class CrossModalEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=512, dropout=0.1, activation='relu', self_attn=None):
        super(CrossModalEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_head, dropout) if self_attn is None else self_attn
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        

        self.activation = _get_activation_fn(activation)

    def forward(self, src, mask, img_feat):
        src2 = self.self_attn(src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.cross_attn(src, img_feat, mask)
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class CrossModalEncoder(nn.Module):
    def __init__(self, config:Config, layer, n_layer=2, self_attn=None):
        super(CrossModalEncoder, self).__init__()
        self.n_layer = n_layer
        self.d_model = config.d_model
        self.n_head = config.nhead
        self.layers = nn.ModuleList([layer(d_model=self.d_model, n_head=self.n_head, self_attn=self_attn) for _ in range(self.n_layer)])

    def forward(self, seq_feat, mask, img_feat):
        output = seq_feat
        for mod in self.layers:
            output = mod(src=output, mask=mask, img_feat=img_feat)
        
        return output



class TransformerDecoder(nn.Module):
    def __init__(self, config, transformer_layer):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([transformer_layer(config.d_model, config.nhead) for _ in range(config.n_layer)])

    def forward(self, inputs, mask, memory):
        outputs = inputs

        for mod in self.layers:
            outputs = mod(tgt=outputs, memory=memory, mask=mask)

        return outputs


class CADFCN(nn.Module):
    def __init__(self, config:Config):
        super(CADFCN, self).__init__()
        self.arg_num = config.arg_num
        self.arg_dim = config.arg_dim + 1

        self.command_fcn = nn.Linear(config.d_model, config.command_type_num)
        self.args_fcn = nn.Linear(config.d_model, config.arg_num * self.arg_dim)

    def forward(self, inputs):
        N, S, _ = inputs.shape

        command_logits = self.command_fcn(inputs)    # Shape [S, N, n_commands]
        args_logits = self.args_fcn(inputs)         # Shape [N, S, arg_num * arg_dim]
        args_logits = args_logits.reshape(N, S, self.arg_num, self.arg_dim)    # Shape [S, N, arg_num, arg_dim]
        
        return command_logits, args_logits


class VAE(nn.Module):
    def __init__(self, cfg: Config):
        super(VAE, self).__init__()

        self.enc_mu_fcn = nn.Linear(cfg.d_model, cfg.d_model)
        self.enc_sigma_fcn = nn.Linear(cfg.d_model, cfg.d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, z):
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.)
        z = mu + sigma * torch.randn_like(sigma)

        return z, mu, logsigma


class DeepCAD(nn.Module):
    def __init__(self, config):
        super(DeepCAD, self).__init__()
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

        # self.vae = VAE(config)
        # self.use_vae = config.use_vae


    def train_mode(self):
        self.train()
        self.is_train = True

    def predict_mode(self):
        self.eval()
        self.is_train = False

    def forward(self, enc_command, enc_args, enc_mask, dec_command=None, dec_args=None, dec_mask=None, z=None, mode='AE'):
        if mode != 'DE':    # 如果decoder模式则不需要encoder
            enc_emb = self.encoder_embedding(enc_command, enc_args)
            enc_attn_mask = _make_attn_mask(mask=enc_mask).to(self._device)
            memory = self.encoder(inputs=enc_emb, mask=enc_attn_mask)
            z = torch.sum(memory * enc_mask.unsqueeze(2), dim=1, keepdim=True) / torch.sum(enc_mask, dim=1, keepdim=True).unsqueeze(2)
            # if self.use_vae:
            #     z, mu, sigma = self.vae(z)

        if mode == 'EN':        # encoder模式，仅使用encoder
            return z
        if mode == 'AE' or mode == 'DE':
            # _dec_command = dec_command[:, :-1]
            # _dec_args = dec_args[:, :-1]
            # _dec_mask = dec_mask[:, :-1]

            # dec_self_attn_massk = _make_decode_attn_mask(_dec_mask).to(self._device)
        # dec_emb = self.decoder_embedding(_dec_command, _dec_args)
        # dec_path_rep = self.decoder(inputs=dec_emb, mask=dec_self_attn_mask, memory=memory[:, :-1])
            dec_emb = self.const_embedding(z)[:, :-1]
            dec_path_rep = self.decoder(inputs=dec_emb, mask=None, memory=z)  # 传入隐向量z和decoder交互

        command_logits, args_logits = self.fcn(dec_path_rep)


        return command_logits, args_logits


class CMCAD(nn.Module):
    def __init__(self, config:Config):
        super(CMCAD, self).__init__()
        self._config = config
        self._device = config.device

        # 图像编码器
        self.img_encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Linear(in_features, config.d_model * config.max_path)

        # 序列编码器
        self.seq_encoder_emb = CADEmbedding(config=config, need_pos=True)
        self.self_attn = None # MultiHeadSelfAttention(config.d_model, config.nhead, config.dropout)   # 共用同一个self_attn，共享参数
        self.seq_encoder = TransformerEncoder(config=config, transformer_layer=TransformerEncoderLayer, n_layer=2, self_attn=self.self_attn)
        
        # 模态融合编码器
        self.cm_encoder = CrossModalEncoder(config=config, layer=CrossModalEncoderLayer, self_attn=self.self_attn)

        # 解码器
        self.decoder_embedding = CADEmbedding(config=config, need_pos=True)
        self.const_embedding = ConstEmbedding(config)
        # self.decoder = TransformerDecoder(config=config, transformer_layer=TransformerDecoderLayer)
        self.decoder = TransformerDecoder(config=config, transformer_layer=TransformerDecoderLayerGlobalImproved)

        self.fcn = CADFCN(config=config)

        self.is_train = True

        self.use_vae = config.use_vae
        if self.use_vae:
            self.vae = VAE(config)

    def train_mode(self):
        self.train()
        self.is_train = True

    def predict_mode(self):
        self.eval()
        self.is_train = False

    def forward(self, enc_command, enc_args, enc_mask, img, dec_command=None, dec_args=None, dec_mask=None, z=None, mode='AE'):
        if mode == 'AE' or mode == 'EN':
            img_encoder_output = self.img_encoder(img).resize(enc_command.shape[0], enc_command.shape[1], 256)

            enc_emb = self.seq_encoder_emb(enc_command, enc_args)
            enc_attn_mask = _make_attn_mask(mask=enc_mask).to(self._device)
            seq_encoder_output = self.seq_encoder(inputs=enc_emb, mask=enc_attn_mask)
            
            memory = self.cm_encoder(seq_encoder_output, enc_attn_mask, img_encoder_output)
            z = torch.sum(memory * enc_mask.unsqueeze(2), dim=1, keepdim=True) / torch.sum(enc_mask, dim=1, keepdim=True).unsqueeze(2)

            # 使用VAE
            if self.use_vae:
                z, mu, sigma = self.vae(z)

        if mode == 'EN':        # encoder模式，仅使用encoder
            return z

        if mode == 'AE' or mode == 'DE':
            # _dec_command = dec_command[:, :-1]
            # _dec_args = dec_args[:, :-1]
            # _dec_mask = dec_mask[:, :-1]

            # dec_self_attn_mask = _make_decode_attn_mask(_dec_mask).to(self._device)
        # dec_emb = self.decoder_embedding(_dec_command, _dec_args)
        # dec_path_rep = self.decoder(inputs=dec_emb, mask=dec_self_attn_mask, memory=memory[:, :-1])
            dec_emb = self.const_embedding(z)[:, :-1]
            dec_path_rep = self.decoder(inputs=dec_emb, mask=None, memory=z)  # 传入隐向量z和decoder交互

            command_logits, args_logits = self.fcn(dec_path_rep)
            return command_logits, args_logits


"""
    消融实验，将模态融合部分更换为直接相加的Add方法， 拼接后过MLP的Concatenate方法
"""
class CMCAD_Ablation(nn.Module):
    def __init__(self, config: Config):
        super(CMCAD_Ablation, self).__init__()
        self._config = config
        self._device = config.device

        # 图像编码器
        self.img_encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Linear(in_features, config.d_model)

        # 序列编码器
        self.seq_encoder_emb = CADEmbedding(config=config, need_pos=True)
        self.self_attn = None  # MultiHeadSelfAttention(config.d_model, config.nhead, config.dropout)   # 共用同一个self_attn，共享参数
        self.seq_encoder = TransformerEncoder(config=config, transformer_layer=TransformerEncoderLayer, n_layer=4,
                                              self_attn=self.self_attn)
        # 拼接融合模态特征后接线性层
        # self.linear = nn.Linear(config.d_model * 2, config.d_model)
        # 解码器
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

    def forward(self, enc_command, enc_args, enc_mask, img, dec_command=None, dec_args=None, dec_mask=None, z=None,
                mode='AE'):
        if mode == 'AE' or mode == 'EN':
            img_encoder_output = self.img_encoder(img).unsqueeze(1)

            enc_emb = self.seq_encoder_emb(enc_command, enc_args)
            enc_attn_mask = _make_attn_mask(mask=enc_mask).to(self._device)
            seq_encoder_output = self.seq_encoder(inputs=enc_emb, mask=enc_attn_mask)
            # memory = self.cm_encoder(seq_encoder_output, enc_attn_mask, img_encoder_output)
            z = torch.sum(seq_encoder_output * enc_mask.unsqueeze(2), dim=1, keepdim=True) / torch.sum(enc_mask, dim=1,
                                                                                           keepdim=True).unsqueeze(2)
            """相加融合模态特征"""
            z = z + img_encoder_output
            """拼接模态特征"""
            # z = self.linear(torch.cat((img_encoder_output, z), dim=2))


        if mode == 'EN':  # encoder模式，仅使用encoder
            return z

        if mode == 'AE' or mode == 'DE':
            # _dec_command = dec_command[:, :-1]
            # _dec_args = dec_args[:, :-1]
            # _dec_mask = dec_mask[:, :-1]

            # dec_self_attn_mask = _make_decode_attn_mask(_dec_mask).to(self._device)
            # dec_emb = self.decoder_embedding(_dec_command, _dec_args)
            # dec_path_rep = self.decoder(inputs=dec_emb, mask=dec_self_attn_mask, memory=memory[:, :-1])
            dec_emb = self.const_embedding(z)[:, :-1]
            dec_path_rep = self.decoder(inputs=dec_emb, mask=None, memory=z)  # 传入隐向量z和decoder交互

            command_logits, args_logits = self.fcn(dec_path_rep)
            return command_logits, args_logits