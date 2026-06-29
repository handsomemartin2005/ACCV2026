# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Transformer modules."""

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

try:
    from mmcv.cnn import build_norm_layer
except ImportError:
    def build_norm_layer(cfg, num_features, postfix=''):
        norm_type = (cfg or {}).get('type', 'LN')
        if norm_type in {'LN', 'LayerNorm'}:
            return f'ln{postfix}', nn.LayerNorm(num_features)
        if norm_type in {'BN', 'BN1d', 'BatchNorm1d'}:
            return f'bn{postfix}', nn.BatchNorm1d(num_features)
        if norm_type in {'BN2d', 'BatchNorm2d'}:
            return f'bn{postfix}', nn.BatchNorm2d(num_features)
        if norm_type in {'GN', 'GroupNorm'}:
            return f'gn{postfix}', nn.GroupNorm((cfg or {}).get('num_groups', 32), num_features)
        raise KeyError(f'Unsupported norm type without mmcv: {norm_type}')

__all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
           'DeformableTransformerDecoder', 'GCN_DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer',
           'MSDeformAttn', 'MLP')


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes (memory_spatial_shapes)
            bbox_head, # bbox_head
            score_head, # score_head
            pos_mlp, # query_pos_head
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class GCN_DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 eval_idx=-1,
                 num_classes=17,

                 # 图嵌入超参数
                 emb_channel=768,
                 norm='GN',
                 stop_grad=False,
                 fusion_mode='dot',
                 emb_mode='label',
                 gcn_num=2,
                 gcn_bias=False,
                 gcn_act='leakyrelu',  # relu
                 cls_use_gcn=True,
                 cls_adj_mode='A_img_lvl_eye',
                 reg_use_gcn=True,
                 reg_adj_mode='A_img_lvl_eye',
                 adj_path='/home/robot/Projects/RTDETR-main/dataset/coco/adj_with_embedding/adj_matrix.pkl',
                 emb_labels_path='/home/robot/Projects/RTDETR-main/dataset/coco_uav/adj_with_embedding/labels_embedding.pkl',
                 emb_words_path='/home/robot/Projects/RTDETR-main/dataset/coco_uav/adj_with_embedding/words_embedding.pkl',
                 ):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_classes = num_classes

        # --------------------------------------------------------- #
        # 整体图卷积层(GCN)的设置
        self.emb_channel = emb_channel
        self.norm = norm
        self.stop_grad = stop_grad
        self.fusion_mode = fusion_mode
        self.emb_mode = emb_mode
        self.num_gcns = gcn_num
        self.gcn_bias = gcn_bias
        self.gcn_act = gcn_act
        self.adj_path = adj_path
        self.emb_labels_path = emb_labels_path
        self.emb_words_path = emb_words_path

        # (semantic guidance module) SGM模块的设置
        self.cls_gcn = cls_use_gcn
        self.cls_adj_mode = cls_adj_mode

        # (postional guidance module) PGM模块的设置
        self.reg_gcn = reg_use_gcn
        self.reg_adj_mode = reg_adj_mode

        # initialize the adjacency matrix 初始化邻接矩阵
        if self.cls_adj_mode == 'random' or self.reg_adj_mode == 'random':
            self.cls_A = torch.randn(self.num_classes, self.num_classes)
            self.reg_A = torch.randn(self.num_classes, self.num_classes)
        elif self.cls_adj_mode == 'ones' or self.reg_adj_mode == 'ones':
            self.cls_A = torch.ones(self.num_classes, self.num_classes)
            self.reg_A = torch.ones(self.num_classes, self.num_classes)
        elif self.cls_adj_mode == 'learn' or self.reg_adj_mode == 'learn':
            self.cls_A = nn.Parameter(torch.randn(self.num_classes, self.num_classes))
            self.reg_A = nn.Parameter(torch.randn(self.num_classes, self.num_classes))
        else:
            if Path(self.adj_path).exists():
                with open(self.adj_path, 'rb') as file:
                    adjacency_matrix = torch.load(file)
            else:
                adjacency_matrix = torch.eye(self.num_classes)
            self.reg_A = adjacency_matrix.float()
            self.cls_A = adjacency_matrix.float()

        # 加载词嵌入 (load word embeddings)
        if self.cls_gcn:
            if self.emb_mode == 'label':
                if Path(self.emb_labels_path).exists():
                    with open(self.emb_labels_path, 'rb') as file:
                        word_embeddings_info = torch.load(file)
                    self.cls_word_embeddings = word_embeddings_info.float()
                else:
                    self.cls_word_embeddings = torch.randn(self.num_classes, self.emb_channel)

            elif self.emb_mode == 'word':
                if Path(self.emb_words_path).exists():
                    with open(self.emb_words_path, 'rb') as file:
                        word_embeddings_info = torch.load(file)
                    self.cls_word_embeddings = word_embeddings_info.float()
                else:
                    self.cls_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'random':
                self.cls_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'ones':
                self.cls_word_embeddings = torch.ones(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'learn':
                self.cls_word_embeddings = nn.Parameter(torch.randn(self.num_classes, self.emb_channel))

        if self.reg_gcn:
            if self.emb_mode == 'label':
                if Path(self.emb_labels_path).exists():
                    with open(self.emb_labels_path, 'rb') as file:
                        word_embeddings_info = torch.load(file)
                    self.reg_word_embeddings = word_embeddings_info.float()
                else:
                    self.reg_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'word':
                if Path(self.emb_words_path).exists():
                    with open(self.emb_words_path, 'rb') as file:
                        word_embeddings_info = torch.load(file)
                    self.reg_word_embeddings = word_embeddings_info.float()
                else:
                    self.reg_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'random':
                self.reg_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'ones':
                self.reg_word_embeddings = torch.ones(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'learn':
                self.reg_word_embeddings = nn.Parameter(torch.randn(self.num_classes, self.emb_channel))

        if self.gcn_act == 'leakyrelu':
            self.gcn_relu = nn.LeakyReLU(inplace=True)
        elif self.gcn_act == 'relu':
            self.gcn_relu = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)

        # if self.cls_gcn or self.reg_gcn:
        if self.cls_gcn:
            if self.norm == 'GN':
                norm_cfg = dict(type='GN', num_groups=8, requires_grad=True)
                self.cls_gcn_norm = build_norm_layer(norm_cfg, self.hidden_dim)[1]
            elif self.norm == 'BN':
                self.cls_gcn_norm = nn.BatchNorm1d(self.hidden_dim)
            elif self.norm == 'LN':
                self.cls_gcn_norm = nn.LayerNorm(self.hidden_dim)

        if self.reg_gcn:
            if self.norm == 'GN':
                norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
                self.reg_gcn_norm = build_norm_layer(norm_cfg, 4)[1]
            elif self.norm == 'BN':
                self.reg_gcn_norm = nn.BatchNorm1d(4)
            elif self.norm == 'LN':
                self.reg_gcn_norm = nn.LayerNorm(4)

        if self.cls_gcn:
            # visual embedding
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.map_fcs = nn.ModuleList()
            self.map_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))  # 输入map为fpn的输出
            # self.map_fcs.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim)) # 输入map为backbone的输出

            self.gcns_cls = self._add_gcns_cls(self.num_gcns, self.emb_channel, self.gcn_bias)
            self.cls_gcn_conv1d = nn.Conv1d(self.hidden_dim, self.hidden_dim, self.num_classes)

        if self.reg_gcn:
            # position embedding
            self.pos_emb_fcs = nn.ModuleList()
            self.pos_emb_fcs.append(nn.Linear(4, 64))
            self.pos_emb_fcs.append(nn.Linear(64, self.num_classes))

            self.gcns_reg_enh = self._add_gcns_reg(self.num_gcns, self.emb_channel, self.gcn_bias)

        # if self.fusion_mode == 'cat':
        #     if self.cls_gcn:
        #         self.cls_fusion_cat_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        #     if self.reg_gcn:
        #         self.reg_fusion_cat_fc = nn.Linear(self.hidden_dim * 2, 4)

    def fusion_w_dim_process(self, targets, gcn_feat, mode='sum'):
        """fusion two tensors
        Args:
            targets (tensor): [B * R, D]  # B=bs, R = num_rois(head)
            gcn_feat (tensor): [B, D]
            # roi_bs_ids: [B * R]
        Returns:
            out (tensor): [B * R, D] or [B * R, 2D]
        """
        # 设 bs = 2, roi = 450, D = 256 则有
        # targets = (900, 256)； gcn_feat = (2, 256)；

        bsR = targets.size(0)  # 900
        bs = gcn_feat.size(0)  # 2
        D = gcn_feat.size(1)  # 256

        if bsR % bs == 0:
            gcn_feat = gcn_feat.reshape(bs, 1, D)  # (2, 1, 256)
            targets = targets.reshape(bs, -1, D)  # (2, 450, 256)

            if mode == 'sum':
                targets = targets + gcn_feat  # (2, 450, 256)
            elif mode == 'dot':
                targets = targets * gcn_feat  # (2, 450, 256)
            # elif mode == 'cat':
            #     # (2, 256) -> (2, 1, 256) -> (2, 450, 256)
            #     gcn_feat = gcn_feat.reshape(bs, -1, D).repeat(1, int(bsR / bs), 1)  # (2, 450, 256)
            #     targets = torch.cat((targets, gcn_feat), dim=-1)  # (2, 450, 512)
            # (2, 450, 512) -> (900, 512) or (2, 450, 256) -> (900, 256)
            targets = targets.reshape(bsR, -1)

        return targets

    def _add_gcns_cls(self, num_gcns, in_channels, use_bias=True):
        gcns = nn.ModuleList()
        for i in range(num_gcns):
            in_channels = (in_channels if i == 0 else self.hidden_dim)
            gcns.append(GraphConvolution(in_channels, self.hidden_dim, use_bias))
        # gcns.append(GraphConvolution(self.hidden_dim, 192, use_bias))
        return gcns

    def _add_gcns_reg(self, num_gcns, in_channels, use_bias=True):
        gcns = nn.ModuleList()
        for i in range(num_gcns - 1):
            in_channels = (in_channels if i == 0 else self.hidden_dim)
            gcns.append(GraphConvolution(in_channels, self.hidden_dim, use_bias))
        gcns.append(GraphConvolution(self.hidden_dim, 4, use_bias))
        return gcns

    def conv1d_w_dim_process(self, conv1d, inp):
        """
        conv1d with dimension proessing

        Args:
            conv1d (nnmodule), act (nnmodule)
            inp (tensor): [C, D]
        Returns:
            out (tensor): [1, D]
        """
        inp = inp.unsqueeze(0).permute(0, 2, 1)
        out = self.relu(conv1d(inp))
        out = out.permute(0, 2, 1).squeeze()
        return out

    def fusion(self, x1, x2, factor, mode='line'):
        """fusion two tensors with same dimension

        Args:
            x1 (tensor): [B*R, D]
            x2 (tensor): [B*R, D]
        Returns:
            out (tensor): [B*R, D] or [B*R, 2D]
        """
        if mode == 'sum':
            out = x1 + x2
        elif mode == 'dot':
            out = x1 * x2
        elif mode == 'line':
            out = x1 + factor * x1 * x2
        return out



    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            backbone_feats,
            feats,  # image features
            shapes,  # feature shapes (memory_spatial_shapes)
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None
    ):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        # get device -> cpu or gpu
        gpu_device = output.device
        bs = output.shape[0]

        # ---------------------------- cls_gcn_embedding process 类别图嵌入过程 --------------------------------- #
        if self.stop_grad:
            feat = backbone_feats.detach()
        else:
            feat = backbone_feats

        if self.cls_gcn:
            """
               Step1: 利用全局平均池化对 C5 特征进行池化操作 (bs, 256, W, H) -> (bs, 256, 1, 1)
               Step2: 利用 flatten 指令对特征图进行压缩 -> (bs, 256, 1, 1) -> (bs, 256)
               Step3: 利用 Fcs 指令对压缩图进行维度规整-> (bs, 256) -> (bs, out_dim)
            """
            # (bs, 192, W, H) -> (bs, 192, 1, 1)
            feat = self.global_pool(feat)
            # (bs, 192, 1, 1) -> (bs, 192)
            feat = feat.flatten(1)
            # 采用 FCs 对特征输出维度进行归整
            for fc in self.map_fcs:
                feat = self.relu(fc(feat))  # (bs, 256)
            """
               将邻接矩阵嵌入 GCN 的过程: emb -> gcns -> norm -> conv1d
            """
            # 选择类别邻接矩阵 (cls_A) 的类型
            if self.cls_adj_mode == 'learn':
                adj = self.cls_A.to(gpu_device) # (17, 17)
            else:
                adj = self.cls_A.detach().to(gpu_device)
            # 定义 Word_embedding GCN 节点特征矩阵
            cls_gcn_fea = self.cls_word_embeddings.to(gpu_device)  # (17, 768)

            # 将 cls_gcn_fea 和 adj 作为输入 GCN网络 得到图知识嵌入特征输出
            for gcn in self.gcns_cls:
                cls_gcn_fea = self.gcn_relu(gcn(cls_gcn_fea, adj))  # (17, 256)

            # 将图知识嵌入输出特征 cls_gcn_fea 重塑为 (17, 256), 用于归一化 1-D Conv 输入
            cls_gcn_fea = cls_gcn_fea.reshape(self.num_classes, self.hidden_dim)  # (17, 256)


            # 利用 批标准化(BN)操作
            if self.norm:
                cls_gcn_fea = self.cls_gcn_norm(cls_gcn_fea)


            # 融合带有视觉信息的输入图像和类关系的特征： [bs, 256] + [1, 256] -> [bs, 256]
            # cls_gcn_conv1d：定义的 1D-Conv 模块
            cls_gcn_conv1d_feat = self.conv1d_w_dim_process(self.cls_gcn_conv1d, cls_gcn_fea)  # size = [256]
            # 利用 GCN 输出 和 C5 进行融合 -> 增强 C5 特征
            cls_enhance_feat = cls_gcn_conv1d_feat + feat  # [bs, 256]

        if self.cls_gcn:
            """
               cls_enhance_feat: 增强后的 C5 特征
            """
            # output = (B, R, 256) & [B, 256]-> (B * R, 256)
            output = self.fusion_w_dim_process(output.reshape(-1, self.hidden_dim), cls_enhance_feat, self.fusion_mode)
            # (B * R, 256) -> (B, R, 256)
            output = output.reshape(bs, -1, self.hidden_dim)
            # if self.fusion_mode == 'cat':
            #     output = self.relu(self.cls_fusion_cat_fc(output))
            #     output = output.reshape(bs, -1, self.hidden_dim * 2)

            # ---------------------------- pos_gcn_embedding process 位置图嵌入过程 --------------------------------- #

            if self.reg_gcn:
                # ref_points_detach = (B, R, 4) -> (B * R, 4)
                relative_ref_points_detach = refer_bbox.reshape(-1, 4)  # (B * R, 4)
                for fc in self.pos_emb_fcs:
                    relative_ref_points_detach = self.relu(fc(relative_ref_points_detach))  # (B * R, 17)

                # emb -> gcns -> norm, output dim [17, 256]
                if self.reg_adj_mode == 'learn':
                    adj = self.reg_A.to(gpu_device)
                else:
                    adj = self.reg_A.detach().to(gpu_device)
                reg_gcn_enh_fea = self.reg_word_embeddings.to(gpu_device)

                for gcn in self.gcns_reg_enh:
                    reg_gcn_enh_fea = self.gcn_relu(gcn(reg_gcn_enh_fea, adj))  # [17, 4]

                reg_gcn_enh_fea = reg_gcn_enh_fea.reshape(self.num_classes, 4)  # [17, 4]

                if self.norm:
                    reg_gcn_enh_fea = self.reg_gcn_norm(reg_gcn_enh_fea)

                # [B * R, 17] x [17, 4] -> [B * R, 4]
                # fuse the features from (input image with sptial information) and (class relation)
                reg_enhance_feat = torch.matmul(relative_ref_points_detach, reg_gcn_enh_fea)  # [B * R, 4]

            if self.reg_gcn:
                # (B * R, 4) & (B * R, 4)
                refer_bbox = self.fusion(refer_bbox.reshape(-1, 4), reg_enhance_feat,
                                                0.05, self.fusion_mode)
                refer_bbox = refer_bbox.reshape(bs, -1, 4)
                # if self.fusion_mode == 'cat':
                #     ref_points_detach = self.relu(self.reg_fusion_cat_fc(reg_enhance_feat))  # (B * R, 4)
                #     ref_points_detach = ref_points_detach.reshape(bs, -1, 4)

            # ---------------------------- *********************************** --------------------------------- #

        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
