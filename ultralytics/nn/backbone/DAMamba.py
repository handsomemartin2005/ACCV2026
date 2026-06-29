import logging
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_
from typing import Sequence
from einops import repeat

try:
    from mmcls.plain_mamba_dev.models.plain_mamba.adapter_modules import (
        SpatialPriorModule, InteractionBlock, get_reference_points, MSDeformAttn)
    from mmcls.models.utils import resize_pos_embed, to_2tuple
    from mmcls.models.backbones.base_backbone import BaseBackbone
    from mmcls.plain_mamba_dev.models.modules.patch_embed import ConvPatchEmbed
except ImportError:
    class BaseBackbone(nn.Module):
        pass

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    def resize_pos_embed(pos_embed, old_resolution, new_resolution, mode='bicubic', num_extra_tokens=0):
        if old_resolution == new_resolution:
            return pos_embed
        extra_tokens = pos_embed[:, :num_extra_tokens] if num_extra_tokens else pos_embed[:, :0]
        pos_tokens = pos_embed[:, num_extra_tokens:]
        c = pos_tokens.shape[-1]
        pos_tokens = pos_tokens.reshape(1, old_resolution[0], old_resolution[1], c).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=new_resolution, mode=mode, align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_resolution[0] * new_resolution[1], c)
        return torch.cat((extra_tokens, pos_tokens), dim=1) if num_extra_tokens else pos_tokens

    class ConvPatchEmbed(nn.Module):
        def __init__(self, in_channels=3, input_size=224, embed_dims=192, num_convs=1, patch_size=16, stride=16):
            super().__init__()
            input_size = to_2tuple(input_size)
            self.init_out_size = (input_size[0] // stride, input_size[1] // stride)
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=stride),
                nn.BatchNorm2d(embed_dims),
                nn.GELU()
            )

        def forward(self, x):
            x = self.proj(x)
            h, w = x.shape[-2:]
            return x.flatten(2).transpose(1, 2).contiguous(), (h, w)

    def _flatten_feature(x):
        return x.flatten(2).transpose(1, 2).contiguous()

    class SpatialPriorModule(nn.Module):
        def __init__(self, inplanes=64, embed_dim=192):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, inplanes, 3, 2, 1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.GELU(),
                nn.Conv2d(inplanes, embed_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )

        def forward(self, x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            return c1, _flatten_feature(c2), _flatten_feature(c3), _flatten_feature(c4)

    class _IdentityInjector(nn.Module):
        def forward(self, query, **kwargs):
            return query

    class _IdentityExtractor(nn.Module):
        def forward(self, query, **kwargs):
            return query

    class InteractionBlock(nn.Module):
        def __init__(self, *args, extra_extractor=False, **kwargs):
            super().__init__()
            self.injector = _IdentityInjector()
            self.extractor = _IdentityExtractor()
            self.extra_extractors = nn.ModuleList([_IdentityExtractor()]) if extra_extractor else None

    def get_reference_points(spatial_shapes, device):
        refs = []
        for h, w in spatial_shapes:
            y, x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device) / h,
                torch.linspace(0.5, w - 0.5, w, device=device) / w,
                indexing='ij')
            refs.append(torch.stack((x, y), -1).reshape(-1, 2))
        return torch.cat(refs, 0)[None]

    class MSDeformAttn(nn.Module):
        def _reset_parameters(self):
            return None

try:
    from mmcv.cnn import build_norm_layer
    from mmcv.cnn.bricks.transformer import build_dropout
    from mmcv.runner.base_module import ModuleList
except ImportError:
    ModuleList = nn.ModuleList

    def build_norm_layer(cfg, num_features, postfix=''):
        norm_type = (cfg or {}).get('type', 'LN')
        if norm_type in {'LN', 'LayerNorm'}:
            return f'ln{postfix}', nn.LayerNorm(num_features)
        if norm_type in {'BN', 'BN2d', 'BatchNorm2d'}:
            return f'bn{postfix}', nn.BatchNorm2d(num_features)
        raise KeyError(f'Unsupported norm type without mmcv: {norm_type}')

    def build_dropout(cfg):
        drop_prob = (cfg or {}).get('drop_prob', 0.)
        drop_type = (cfg or {}).get('type', 'Dropout')
        if drop_type == 'DropPath':
            return DropPath(drop_prob) if drop_prob > 0 else nn.Identity()
        return nn.Dropout(drop_prob)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    def selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=None, delta_softplus=True,
                          return_last_state=False):
        y = u * D[None, :, None].to(dtype=u.dtype, device=u.device)
        return (y, None) if return_last_state else y

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm
except ImportError:
    RMSNorm = nn.LayerNorm

_logger = logging.getLogger(__name__)

__all__ = ['mamba_adapters', 'sch_mamba_adapters']

class PlainMamba2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        conv_bias=True,
        bias=False,
        init_layer_scale=None,
        default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4
        if default_hw_shape is not None:
            orders, inverse_orders, directions = self.get_permute_order(default_hw_shape)
            (
                self.default_permute_order,
                self.default_permute_order_inverse,
                self.default_direction
            ) = orders, inverse_orders, directions

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        padding = int(conv_size // 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=(conv_size, conv_size),
            stride=(1, 1),
            padding=(padding, padding),
            groups=self.d_inner
        )

        self.activation = "silu"
        self.action = nn.SiLU()  # SiLU

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions+1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)


    def get_permute_order(self, hw_shape):
        if self.default_permute_order is not None:
            if hw_shape[0] == self.default_hw_shape[0] and hw_shape[1] == self.default_hw_shape[1]:
                return self.default_permute_order, self.default_permute_order_inverse, self.default_direction
        H, W = hw_shape
        L = H * W

        # [start, right, left, up, down] [0, 1, 2, 3, 4]

        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        j_d = "right"
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"

            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        o2 = []
        d2 = []
        o2_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H-1, W-1
            j_d = "left"
        else:
            i, j = H-1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d2.append(1)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d2.append(2)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "right"
        d2 = [0] + d2[:-1]

        o3 = []
        d3 = []
        o3_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o3_inverse[idx] = len(o3)
            o3.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d3.append(4)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d3.append(3)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "down"
        d3 = [0] + d3[:-1]

        o4 = []
        d4 = []
        o4_inverse = [-1 for _ in range(L)]

        if W % 2 == 1:
            i, j = H - 1, W - 1
            i_d = "up"
        else:
            i, j = 0, W - 1
            i_d = "down"
        while j > -1:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o4_inverse[idx] = len(o4)
            o4.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d4.append(4)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d4.append(3)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "down"
        d4 = [0] + d4[:-1]

        o1 = tuple(o1)
        d1 = tuple(d1)
        o1_inverse = tuple(o1_inverse)

        o2 = tuple(o2)
        d2 = tuple(d2)
        o2_inverse = tuple(o2_inverse)

        o3 = tuple(o3)
        d3 = tuple(d3)
        o3_inverse = tuple(o3_inverse)

        o4 = tuple(o4)
        d4 = tuple(d4)
        o4_inverse = tuple(o4_inverse)

        return (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4)

    def forward(self, x, hw_shape):

        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None

        xz = self.in_proj(x)  # [B, L, 2 * E]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1)
        x = x.clone()
        z = z.clone()

        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.action(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()  # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous()  # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous()  # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.get_permute_order(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]  # each [L, d_state]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]

        sum_y = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,).permute(0, 2, 1)[:, inv_o, :] for o, inv_o, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        liner_y = torch.mul(sum(sum_y), self.action(z))

        out = self.out_proj(liner_y)

        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

class PlainMambaLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm,
        with_dwconv,
        drop_path_rate,
        mamba_cfg,
    ):
        super(PlainMambaLayer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})

        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        self.mamba = PlainMamba2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, x, hw_shape):
        mixed_x = self.drop_path(self.mamba(self.norm(x), hw_shape)) # (1, 1600, 192)
        mixed_x = mixed_x + x  # (1, 1600, 192)

        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.dw(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)
        # print(mixed_x.shape)
        return mixed_x


class InteractionBlock_PlainMamba(InteractionBlock):
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, patch_resolution):
        H, W = patch_resolution

        x = self.injector(query=x,
                          reference_points=deform_inputs1[0],
                          feat=c,
                          spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        for idx, blk in enumerate(blocks):
            x = blk(x, patch_resolution)

        c = self.extractor(query=c,
                           reference_points=deform_inputs2[0],
                           feat=x,
                           spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],
                           H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c,
                              reference_points=deform_inputs2[0],
                              feat=x,
                              spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],
                              H=H, W=W)
        return x, c


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs1, deform_inputs2


class PlainMambaAdapter(nn.Module):

    arch_zoo = {
        'L1': {
            'patch_size': 16,
            'embed_dims': 192,
            'num_layers': 12,  # 24
            'num_convs_patch_embed': 1,
            'layers_with_dwconv': [0],  # useful for L1 model
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
        'L2': {
            'patch_size': 16,
            'embed_dims': 384,
            'num_layers': 24,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
        'L3': {
            'patch_size': 16,
            'embed_dims': 448,
            'num_layers': 36,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
    }
    def __init__(self,
                 # PlainMamba
                 in_channels=3,
                 arch='L1',
                 patch_size=16,
                 embed_dims=256,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=[],


                 # Adpater-mamba
                 pretrain_size=224,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 num_classes=17,
                 ):
        super().__init__()

        # --------------------------- PlainMamba --------------------------- #
        self.arch = arch
        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            _layer_cfgs = layer_cfgs
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=pretrain_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.patch_resolution = self.patch_embed.init_out_size
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        _drop_path_rate = drop_path_rate

        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layer_cfgs = _layer_cfgs

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]
        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i]
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            self.layers.append(
                PlainMambaLayer(**_layer_cfg_i)
            )

        # --------------------------- Adapter-mamba --------------------------- #
        self.num_classes = num_classes
        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        # embed_dim = self.embed_dims

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dims))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,embed_dim=self.embed_dims)
        self.interactions = nn.Sequential(*[
            InteractionBlock_PlainMamba(
                dim=self.embed_dims,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=self.drop_path_rate,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                down_stride=16
            )
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2)
        self.ad_norm = nn.BatchNorm2d(self.embed_dims)
        # self.ad_norm2 = nn.BatchNorm2d(self.embed_dims)
        # self.ad_norm3 = nn.BatchNorm2d(self.embed_dims)
        # self.ad_norm4 = nn.BatchNorm2d(self.embed_dims)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.channel = [192, 192, 192, 192]
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)  # s4, s8, s16, s32
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        H, W = patch_resolution
        bs, n, dim = x.shape
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0)
            x = x + pos_embed
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, patch_resolution)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        feas = [c1, c2, c3, c4]
        # Final Norm
        outs = []
        for ith in range(len(feas)):
            norm_fea = self.ad_norm(feas[ith])
            outs.append(norm_fea)

        return outs

class CLIPGuidedSemanticPrototypeLearning(nn.Module):
    """CLIP text/image priors projected into the detector feature space."""

    DEFAULT_CATEGORY_NAMES = (
        'casing base', 'insulator', 'insulator base', 'isoelectric line', 'load bearing cable base',
        'locator bracing base', 'locator bracing base ear', 'locator clamp', 'locator hook', 'locator ring',
        'locator tube connector', 'rectangular locator', 'rotary double ear', 'sleeve double ear',
        'sleeve screw', 'windproof wire', 'windproof wire ring'
    )

    def __init__(self, num_classes=17, embed_dim=192, clip_model='RN50', clip_pretrained='openai',
                 category_names=None, use_clip=True, use_image_prior=True):
        super().__init__()
        self.use_real_clip = False
        self.use_image_prior = use_image_prior
        self.clip = None
        self.tokenizer = None
        self.category_names = tuple(category_names or self.DEFAULT_CATEGORY_NAMES[:num_classes])
        if len(self.category_names) < num_classes:
            self.category_names = self.category_names + tuple(f'class {i + 1}' for i in range(len(self.category_names), num_classes))
        self.category_names = self.category_names[:num_classes]

        clip_dim = 512
        if use_clip:
            try:
                import open_clip
                self.clip, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
                self.tokenizer = open_clip.get_tokenizer(clip_model)
                self.clip.eval()
                for p in self.clip.parameters():
                    p.requires_grad_(False)
                clip_dim = self.clip.text_projection.shape[1] if hasattr(self.clip, 'text_projection') else clip_dim
                prompts = [f'a UAV image of catenary {name}' for name in self.category_names]
                self.register_buffer('clip_text_tokens', self.tokenizer(prompts), persistent=False)
                self.use_real_clip = True
            except Exception as exc:
                _logger.warning('CLIP initialization failed, using learnable semantic prototypes instead: %s', exc)

        self.clip_to_model = nn.Linear(clip_dim, embed_dim)
        self.fallback_category_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.visual_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.register_buffer('clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)

    def _freeze_clip(self):
        if self.clip is not None:
            self.clip.eval()
            for p in self.clip.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_clip()
        return self

    def _encode_text(self, device):
        self._freeze_clip()
        tokens = self.clip_text_tokens.to(device)
        with torch.no_grad():
            text_features = self.clip.encode_text(tokens).float()
        return F.normalize(self.clip_to_model(text_features), dim=-1)

    def _encode_image(self, image):
        self._freeze_clip()
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image = (image.clamp(0, 1) - self.clip_mean.to(image.device, image.dtype)) / self.clip_std.to(image.device, image.dtype)
        with torch.no_grad():
            image_features = self.clip.encode_image(image).float()
        return F.normalize(self.clip_to_model(image_features), dim=-1)

    def forward(self, features, image=None):
        pooled = torch.stack([F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in features], dim=1).mean(1)
        detector_visual_prior = F.normalize(self.visual_proj(pooled), dim=-1)

        if self.use_real_clip and image is not None:
            prototypes = self._encode_text(pooled.device)
            if self.use_image_prior:
                visual_prior = F.normalize(detector_visual_prior + self._encode_image(image), dim=-1)
            else:
                visual_prior = detector_visual_prior
        else:
            prototypes = F.normalize(self.fallback_category_prototypes, dim=-1)
            visual_prior = detector_visual_prior

        semantic_response = visual_prior @ prototypes.t()
        semantic_prior = semantic_response.softmax(dim=-1) @ prototypes
        return semantic_prior, semantic_response


class DynamicTopKScaleRouter(nn.Module):
    """Selects K useful scales from an M-scale candidate feature bank."""

    def __init__(self, embed_dim=192, topk=3, max_scales=8, use_semantic_prior=True,
                 use_high_frequency=True, use_scale_prior=True, use_structure_prior=True):
        super().__init__()
        self.topk = topk
        self.max_scales = max_scales
        self.use_semantic_prior = use_semantic_prior
        self.use_high_frequency = use_high_frequency
        self.use_scale_prior = use_scale_prior
        self.use_structure_prior = use_structure_prior
        hidden_dim = max(embed_dim // 2, 32)
        self.scale_embed = nn.Parameter(torch.randn(max_scales, embed_dim) * 0.02)
        self.structure_prior = nn.Parameter(torch.zeros(embed_dim))
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _high_frequency_score(x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return (x - smooth).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

    @staticmethod
    def _resize_like(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, candidates, semantic_prior):
        scores = []
        b = candidates[0].shape[0]
        for i, feat in enumerate(candidates):
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            semantic = semantic_prior if self.use_semantic_prior else torch.zeros_like(semantic_prior)
            high_freq = self._high_frequency_score(feat) if self.use_high_frequency else feat.new_zeros((b, 1))
            h, w = feat.shape[-2:]
            scale = feat.new_full((b, 1), math.log(max(h * w, 1))) if self.use_scale_prior else feat.new_zeros((b, 1))
            if self.use_scale_prior:
                scale_prior = self.scale_embed[min(i, self.max_scales - 1)].to(feat.device, feat.dtype).expand(b, -1)
            else:
                scale_prior = feat.new_zeros((b, pooled.shape[1]))
            if self.use_structure_prior:
                structure_prior = self.structure_prior.to(feat.device, feat.dtype).expand(b, -1)
            else:
                structure_prior = feat.new_zeros((b, pooled.shape[1]))
            route_in = torch.cat((pooled, semantic, scale_prior, structure_prior, high_freq, scale), dim=1)
            scores.append(self.router(route_in))
        scores = torch.cat(scores, dim=1)
        k = min(max(self.topk, 1), len(candidates))
        # Use a batch-level TopK so all samples in the batch share tensor shapes.
        selected_idx = torch.topk(scores.mean(0), k=k).indices.sort().values.tolist()
        gates = scores.sigmoid()
        soft_weights = scores.softmax(dim=1)
        selected = []
        for i in selected_idx:
            hard_feat = candidates[i] * gates[:, i].view(b, 1, 1, 1)
            if self.training:
                soft_feat = sum(
                    self._resize_like(candidates[j], candidates[i]) *
                    soft_weights[:, j].view(b, 1, 1, 1)
                    for j in range(len(candidates))
                )
                # Straight-through path: hard TopK tensors are used in forward, while the soft path
                # gives the detection loss gradients to every candidate-scale score.
                hard_feat = hard_feat + soft_feat - soft_feat.detach()
            selected.append(hard_feat)
        return selected, scores, selected_idx


class SCHMambaAdapter(nn.Module):
    """SCH-MDETR backbone wrapper: DAMamba features + semantic prior + dynamic Top-K scale routing."""

    def __init__(self, candidate_scales=5, topk=3, num_classes=17, clip_model='RN50', clip_pretrained='openai',
                 use_clip=True, output_scales=3, use_semantic_prior=True, use_high_frequency=True,
                 use_scale_prior=True, use_structure_prior=True, use_guide_semantic=True,
                 use_clip_image=True):
        super().__init__()
        self.base = PlainMambaAdapter(num_classes=num_classes)
        self.candidate_scales = candidate_scales
        self.topk = topk
        self.output_scales = output_scales
        self.use_guide_semantic = use_guide_semantic
        self.embed_dim = self.base.embed_dims
        self.semantic_prior = CLIPGuidedSemanticPrototypeLearning(
            num_classes=num_classes,
            embed_dim=self.embed_dim,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            use_clip=use_clip,
            use_image_prior=use_clip_image)
        self.router = DynamicTopKScaleRouter(
            embed_dim=self.embed_dim,
            topk=topk,
            max_scales=candidate_scales,
            use_semantic_prior=use_semantic_prior,
            use_high_frequency=use_high_frequency,
            use_scale_prior=use_scale_prior,
            use_structure_prior=use_structure_prior)
        self.extra_downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU()
            )
            for _ in range(max(candidate_scales - 4, 0))
        ])
        self.channel = [self.embed_dim for _ in range(output_scales + 1)]

    def _mix_topk_to_targets(self, candidates, route_scores, selected_idx, target_refs):
        b = candidates[0].shape[0]
        topk_weights = route_scores[:, selected_idx].softmax(dim=1)
        all_weights = route_scores.softmax(dim=1)
        outputs = []
        for ref in target_refs:
            hard = sum(
                DynamicTopKScaleRouter._resize_like(candidates[idx], ref) *
                topk_weights[:, j].view(b, 1, 1, 1)
                for j, idx in enumerate(selected_idx)
            )
            if self.training:
                soft = sum(
                    DynamicTopKScaleRouter._resize_like(candidates[j], ref) *
                    all_weights[:, j].view(b, 1, 1, 1)
                    for j in range(len(candidates))
                )
                hard = hard + soft - soft.detach()
            outputs.append(hard)
        return outputs

    def forward(self, x):
        base_feats = self.base(x)
        candidates = list(base_feats)
        feat = base_feats[-1]
        for downsample in self.extra_downsamples:
            feat = downsample(feat)
            candidates.append(feat)
        candidates = candidates[:self.candidate_scales]
        semantic_prior, semantic_response = self.semantic_prior(candidates, x)
        _, route_scores, selected_idx = self.router(candidates, semantic_prior)
        target_refs = list(base_feats[1:1 + self.output_scales])
        selected = self._mix_topk_to_targets(candidates, route_scores, selected_idx, target_refs)
        self.last_route_scores = route_scores.detach()
        self.last_route_gates = route_scores.sigmoid().detach()
        self.last_semantic_response = semantic_response.detach()
        self.last_route_indices = torch.as_tensor(selected_idx, device=route_scores.device)
        self.last_candidate_shapes = [tuple(feat.shape[-2:]) for feat in candidates]
        guide = base_feats[0]
        if self.use_guide_semantic:
            guide = guide + semantic_prior.view(semantic_prior.shape[0], self.embed_dim, 1, 1)
        return [guide] + selected

def mamba_adapters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlainMambaAdapter().to(device)
    return model

def sch_mamba_adapters(candidate_scales=5, topk=3, clip_model='RN50', clip_pretrained='openai', use_clip=True,
                       output_scales=3, use_semantic_prior=True, use_high_frequency=True,
                       use_scale_prior=True, use_structure_prior=True, use_guide_semantic=True,
                       use_clip_image=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCHMambaAdapter(
        candidate_scales=candidate_scales,
        topk=topk,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        use_clip=use_clip,
        output_scales=output_scales,
        use_semantic_prior=use_semantic_prior,
        use_high_frequency=use_high_frequency,
        use_scale_prior=use_scale_prior,
        use_structure_prior=use_structure_prior,
        use_guide_semantic=use_guide_semantic,
        use_clip_image=use_clip_image).to(device)
    return model

if __name__ == '__main__':

    from thop import clever_format, profile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input = torch.rand([1, 3, 640, 640]).cuda()
    model = mamba_adapters()

    # 计算para和FLOPs
    model.eval()

    macs, params = profile(model.to(device), (input,), verbose=False)
    flops, params = clever_format([macs * 2, params], "%.3f")  # 计算MACs -> Flops = 2 * MACs

    print("Total_FLOPs: %s" % (flops))
    print("Total_params: %s" % (params))
