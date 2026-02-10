from dataclasses import dataclass, replace, field
from functools import partial
from typing import Callable, Optional, Union, Tuple, List

import math
import torch
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, DropPath, ClassifierHead, NormMlpClassifierHead
from timm.layers import create_attn, get_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, create_pool2d
from timm.layers import trunc_normal_tf_, to_2tuple, extend_tuple, make_divisible, _assert
from timm.layers import RelPosMlp, RelPosBias, RelPosBiasTf, resize_rel_pos_bias_table
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import named_apply, checkpoint_seq

import math
import torch.nn.functional as F
from .wsfvit_wavelets import DWT_2D_FFT_L2

__all__ = ['wsfVitCfg', 'wsfVitConvCfg', 'wsfVitTransformerCfg', 'wsfVit']

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    mmdet = True
except ImportError:
    mmdet = False

try:
    from mmpose.models.builder import BACKBONES as pose_BACKBONES
    from mmpose.utils import get_root_logger
    mmpose = True
except ImportError:
    mmpose = False


if mmdet or mmpose:
    from mmengine.logging import MMLogger
    from mmengine.model.weight_init import trunc_normal_
                            
    from mmengine.runner.checkpoint import CheckpointLoader
    from mmengine.utils import to_2tuple
    
    classification = False
else:
    classification = True
    from timm.models._registry import register_model, generate_default_cfgs


@dataclass
class wsfVitTransformerCfg:
    dim_head: int = 32
    head_first: bool = False
    expand_ratio: float = 4.0
    expand_first: bool = True
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias_tf'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio: int = 32
    window_size: Optional[Tuple[int, int]] = (7, 7)
    grid_size: Optional[Tuple[int, int]] = (7, 7)
    no_block_attn: bool = False 
    use_nchw_attn: bool = False 
    init_values: Optional[float] = None
    act_layer: str = 'gelu_tanh'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class wsfVitConvCfg:
    block_type: str = 'mbconv'
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    padding: str = 'same'
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu_tanh'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = 1e-3

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
        assert self.block_type in ('mbconv', 'convnext')
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class wsfVitCfg:
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    num_heads: Tuple[int, ...] = (4, 8, 16, 32)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    mlp_ratio: float = None
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'T', 'T')
    stem_width: Union[int, Tuple[int, int]] = 64
    stem_bias: bool = False
    conv_cfg: wsfVitConvCfg = field(default_factory=wsfVitConvCfg)
    transformer_cfg: wsfVitTransformerCfg = field(default_factory=wsfVitTransformerCfg)
    head_hidden_size: int = None
    weight_init: str = 'vit_eff'


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class AttentionCl(nn.Module):
    """ Channels-last multi-head attention (B, ..., C) """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            partition_block: bool = True
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.partition_block = partition_block

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        if self.partition_block: 
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.rel_pos is not None:
                attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        else:
            k = k * self.scale
            attn = k @ v.transpose(-2, -1)
            if self.rel_pos is not None:
                attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ q

        x = x.transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class Downsample2d(nn.Module):
    """ A downsample pooling module supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            padding: str = '',
            bias: bool = True,
    ):
        super().__init__()
        assert pool_type in ('max', 'max2', 'avg', 'avg2')
        if pool_type == 'max':
            self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=padding or 1)
        elif pool_type == 'max2':
            self.pool = create_pool2d('max', 2, padding=padding or 0)  # kernel_size == stride == 2
        elif pool_type == 'avg':
            self.pool = create_pool2d(
                'avg', kernel_size=3, stride=2, count_include_pad=False, padding=padding or 1)
        else:
            self.pool = create_pool2d('avg', 2, padding=padding or 0)

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x

def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class MbConvBlock(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: wsfVitConvCfg = wsfVitConvCfg(),
            drop_path: float = 0.
    ):
        super(MbConvBlock, self).__init__()
        norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
        groups = num_groups(cfg.group_size, mid_chs)

        self.stride = stride
        
        if stride == 2:
            self.shortcut = Downsample2d(
                in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding)
        else:
            self.shortcut = nn.Identity()

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
            stride_pool, dilation_2 = stride, dilation[1]
            # FIXME handle dilation of avg pool
        elif cfg.stride_mode == '1x1':
            # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
        
        if stride_pool > 1:
            self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding)
        else:
            self.down = nn.Identity()
        
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1)
        self.norm1 = norm_act_layer(mid_chs)

        self.conv2_kxk = create_conv2d(
            mid_chs, mid_chs, cfg.kernel_size,
            stride=stride_2, dilation=dilation_2, groups=groups, padding=cfg.padding)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = cfg.attn_act_layer
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        if cfg.attn_early:
            self.se_early = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)
            self.norm2 = norm_act_layer(mid_chs)
            self.se = None
        else:
            self.se_early = None
            self.norm2 = norm_act_layer(mid_chs)
            self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=cfg.output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        
        x = self.down(x)

        x = self.conv1_1x1(x)
        x = self.norm1(x)

        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)

        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
        return x


def intra_subband_local_partition(x, window_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, '')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def intra_subband_local_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def intra_subband_global_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def intra_subband_global_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


def get_rel_pos_cls(cfg: wsfVitTransformerCfg, window_size):
    rel_pos_cls = None
    if cfg.rel_pos_type == 'mlp':
        rel_pos_cls = partial(RelPosMlp, window_size=window_size, hidden_dim=cfg.rel_pos_dim)
    elif cfg.rel_pos_type == 'bias':
        rel_pos_cls = partial(RelPosBias, window_size=window_size)
    elif cfg.rel_pos_type == 'bias_tf':
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
    return rel_pos_cls


class PartitionAttentionCl(nn.Module):

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: wsfVitTransformerCfg = wsfVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps)  # NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)

        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer(dim)
        self.attn = AttentionCl(
            dim,
            dim,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
            partition_block=self.partition_block,
        )
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_block:
            partitioned = intra_subband_local_partition(x, self.partition_size)
        else:
            partitioned = intra_subband_global_partition(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = intra_subband_local_reverse(partitioned, self.partition_size, img_size)
        else:
            x = intra_subband_global_reverse(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    

class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):

        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ChannelAttention(nn.Module):
 
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        k = k.transpose(-1,-2)
        attention = k @ v
        attention = attention.softmax(dim=-1)
        q = q.transpose(-1,-2)
        x = (attention @ q)
        x = x.transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ConvPosEnc(nn.Module):

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x


class wsfVitBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int,
            stride: int = 1,
            conv_cfg: wsfVitConvCfg = wsfVitConvCfg(),
            transformer_cfg: wsfVitTransformerCfg = wsfVitTransformerCfg(),
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            layer: int = 0,
    ):
        super().__init__()
        self.nchw_attn = transformer_cfg.use_nchw_attn

        qkv_bias=True
        ffn=True

        self.dim = dim_out
        self.num_heads = num_heads

        self.conv = MbConvBlock(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)

        attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
        self.attn_block = None if transformer_cfg.no_block_attn else PartitionAttentionCl(**attn_kwargs)

        self.layer = layer
        self.attn_channel = ChannelBlock(dim=self.dim,
                        num_heads=self.num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path,
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,)

        self.attn_grid = PartitionAttentionCl(partition_type='grid', **attn_kwargs)

    def forward(self, x):
        x = self.conv(x)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.attn_block(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.flatten(2).transpose(1,2).contiguous()
        x = self.attn_channel(x, (H, W))
        x = x.reshape(B, H, W, C).contiguous()

        x = self.attn_grid(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class wsfVitStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            num_heads: int,
            stride: int = 2,
            depth: int = 4,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: wsfVitTransformerCfg = wsfVitTransformerCfg(),
            conv_cfg: wsfVitConvCfg = wsfVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
            mlp_ratio: float = 4.,
            layer: int = 0,
    ):
        super().__init__()
        self.grad_checkpointing = False

        block_types = extend_tuple(block_types, depth)
        blocks = []
        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1
 
            blocks += [wsfVitBlock(
                in_chs,
                out_chs,
                num_heads,
                stride=block_stride,
                conv_cfg=conv_cfg,
                transformer_cfg=transformer_cfg,
                drop_path=drop_path[i],
                mlp_ratio=mlp_ratio,
                layer=layer
            )]
            
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        self.dwt_l2 = DWT_2D_FFT_L2(wave='haar')

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]

        self.norm1 = norm_act_layer(in_chs*4)

        self.conv1 = create_conv2d(in_chs*4, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)        
        self.norm2 = norm_act_layer(out_chs[0])
        
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)
        self.norm3 = norm_act_layer(out_chs[1])

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        LL_1, LH_1, HL_1, HH_1 = self.dwt_l2(x)
        x = torch.cat((LL_1, LH_1, HL_1, HH_1), dim=1)
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.norm2(x)

        x = self.conv2(x)
        x = self.norm3(x)
        return x


def cfg_window_size(cfg: wsfVitTransformerCfg, img_size: Tuple[int, int]):
    if cfg.window_size is not None:
        assert cfg.grid_size
        return cfg
    partition_size = img_size[0] // cfg.partition_ratio, img_size[1] // cfg.partition_ratio
    cfg = replace(cfg, window_size=partition_size, grid_size=partition_size)
    return cfg


def _overlay_kwargs(cfg: wsfVitCfg, **kwargs):
    transformer_kwargs = {}
    conv_kwargs = {}
    base_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith('transformer_'):
            transformer_kwargs[k.replace('transformer_', '')] = v
        elif k.startswith('conv_'):
            conv_kwargs[k.replace('conv_', '')] = v
        else:
            base_kwargs[k] = v
    cfg = replace(
        cfg,
        transformer_cfg=replace(cfg.transformer_cfg, **transformer_kwargs),
        conv_cfg=replace(cfg.conv_cfg, **conv_kwargs),
        **base_kwargs
    )
    return cfg


class wsfVit(nn.Module):
    def __init__(
            self,
            cfg: wsfVitCfg = "wsfViT_Tiny",
            fork_feat : bool =False,
            model_name: str = None,
            pretrained_path: str = None,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.2,
            **kwargs,
    ):
        super().__init__()        
        if classification:
            if kwargs:
                cfg = _overlay_kwargs(cfg, **kwargs)
        else:
            logger = MMLogger.get_current_instance()

        self.model_name      = model_name
        if self.model_name is None:
            print(f'No Model Name')
        else:
            cfg = model_cfgs[self.model_name]

        self.fork_feat = fork_feat
        self.pretrained_path = pretrained_path
        img_size = to_2tuple(img_size)

        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = cfg.embed_dim[-1]
        self.num_heads = cfg.num_heads[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []
        
        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
            padding=cfg.conv_cfg.padding,
            bias=cfg.stem_bias,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
        )
        stride = 2
        
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

        self.num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == self.num_stages
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        in_chs = self.stem.out_chs
        stages = []
        for i in range(self.num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            num_heads = cfg.num_heads[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [wsfVitStage(
                in_chs,
                out_chs,
                num_heads,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[i],
                mlp_ratio=cfg.mlp_ratio,
                layer=i,
            )]
            stride *= stage_stride
            in_chs = out_chs
            self.feature_info += [dict(num_chs=out_chs, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        if classification:
            final_norm_layer = partial(get_norm_layer(cfg.transformer_cfg.norm_layer), eps=cfg.transformer_cfg.norm_eps)
            self.norm = nn.Identity()
            self.head_hidden_size = cfg.head_hidden_size
            if self.head_hidden_size:
                self.head = NormMlpClassifierHead(
                    self.num_features,
                    num_classes,
                    hidden_size=self.head_hidden_size,
                    pool_type=global_pool,
                    drop_rate=drop_rate,
                    norm_layer=final_norm_layer,
                )
            else:
                self.norm = final_norm_layer(self.num_features)
                # standard classifier head w/ norm, pooling, fc classifier
                self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        else:
            for i_layer in range(self.num_stages):
                layer = LayerNorm2d(cfg.embed_dim[i_layer])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)
        
    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    def _init_weights__(self, m):

        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out // m.groups
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)  

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)            

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.constant_(m.weight, 1)
        
        if self.fork_feat:    
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                        
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            
    def init_weights(self, scheme=''):
        logger = MMLogger.get_current_instance()

        if self.pretrained_path is None:
            print("********************")
            print(f"prtrained_path is not loaded : {self.pretrained_path}")
            print("********************")

            logger.warn(f'No pre-trained weights for '
                         f'{self.__class__.__name__}, '
                         f'training start from scratch')
            
            self.apply(self._init_weights__)
        else:
            print("********************")
            print("Initiating Weight...")
            
            self.apply(self._init_weights__)
            ckpt = CheckpointLoader.load_checkpoint(
                self.pretrained_path, logger=logger, map_location='cpu')

            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['state_dict']
            else:
                _state_dict = ckpt

            self.load_state_dict(_state_dict, False)
            print("Pretrained Weight is loaded!")
            print("********************")


    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)            

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for stage_index in range(self.num_stages):
            layer = self.stages[stage_index]
            x = layer(x)

            if self.fork_feat:
                norm_layer = getattr(self, f'norm{stage_index}')
                out = norm_layer(x)

                outs.append(out)
        
        if self.fork_feat:
            return outs
        else:
            x = self.norm(x)
            return self.forward_head(x)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.fork_feat:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


def interpolate_rel_pos_bias(state_dict, model):

    print("************************************************")
    print("[Info] Interpolating relative position bias table...")
    for name, param in model.named_parameters():
        if "relative_position_bias_table" in name and name in state_dict:
            print(name)
            old_bias_table = state_dict[name]
            print("[Info] Old bias table shape:", old_bias_table.shape)
            print("[Info] New bias table shape:", param.shape)

            # Check if the shapes are different
            if old_bias_table.shape != param.shape:
                old_n_heads, old_len, _ = old_bias_table.shape
                new_n_heads, new_len, _ = param.shape

                if old_n_heads != new_n_heads:
                    print(f"[Warning] `{name}` #heads mismatch: {old_n_heads} vs. {new_n_heads}. Skipped.")
                    continue
                
                old_bias_table = old_bias_table.unsqueeze(0) # shape: [1, n_heads, old_dim, old_dim]
                new_bias_table = F.interpolate(
                    old_bias_table,
                    size=(new_len, new_len),
                    mode='bicubic',
                    align_corners=False
                )  # shape: [n_heads, 1, new_dim, new_dim]
                
                new_bias_table = new_bias_table.squeeze(0) # shape: [n_heads, new_dim, new_dim]
                print("[Info] Interpolated bias table shape:", new_bias_table.shape)
                state_dict[name] = new_bias_table
            else:
                print(f"[Warning] `{name}` shape {old_bias_table.shape} -> {param.shape} not suitable for 2D interp.")
    print("************************************************")
    return state_dict


def checkpoint_filter_fn(state_dict, model: nn.Module):
    model_state_dict = model.state_dict()
    out_dict = {}
    for k, v in state_dict.items():
        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )

        if k in model_state_dict and v.ndim != model_state_dict[k].ndim and v.numel() == model_state_dict[k].numel():
            # adapt between conv2d / linear layers
            assert v.ndim in (2, 4)
            v = v.reshape(model_state_dict[k].shape)
        out_dict[k] = v
    return out_dict


def _create_wsfvit(variant, cfg_variant=None, pretrained=False, **kwargs):
    
    if cfg_variant is None:
        if variant in model_cfgs:
            cfg_variant = variant
        else:
            cfg_variant = '_'.join(variant.split('_')[:-1])
    return build_model_with_cfg(
        wsfVit, variant, pretrained,
        model_cfg=model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }

default_cfgs = generate_default_cfgs({
    'wsfvit_p1_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, crop_pct=0.875),
    
    'wsfvit_p2_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, crop_pct=0.875), 
    
    'wsfvit_n_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, crop_pct=0.875),
    
    'wsfvit_t_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'wsfvit_t_384.in1k': _cfg(
        file='/path/to/checkpoint', 
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=0.95, crop_mode='squash'
    ),
    
    'wsfvit_s_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'wsfvit_s_384.in1k': _cfg(
        file='/path/to/checkpoint', 
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=0.95, crop_mode='squash'
    ),

    'wsfvit_m_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'wsfvit_m_384.in1k': _cfg(
        file='/path/to/checkpoint', 
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=0.95, crop_mode='squash'
    ),

    'wsfvit_b_224.in1k': _cfg(
        file='/path/to/checkpoint', 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'wsfvit_b_256.in1k': _cfg(
        file='/path/to/checkpoint', 
        input_size=(3, 256, 256), pool_size=(8, 8),
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'wsfvit_b_384.in1k': _cfg(
        file='/path/to/checkpoint', 
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=0.95, crop_mode='squash'
    ),
})

model_cfgs = dict(
        wsfvit_p1_224=wsfVitCfg(
            embed_dim=(32, 64, 128, 256),
            depths=(1, 1, 2, 1),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=32,
            stem_bias=True,
            head_hidden_size=256,
        ),
        wsfvit_p2_224=wsfVitCfg(
            embed_dim=(32, 64, 128, 256),
            depths=(1, 2, 3, 1),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=32,
            stem_bias=True,
            head_hidden_size=256,
        ),
        wsfvit_n_224=wsfVitCfg(
            embed_dim=(32, 64, 128, 256),
            depths=(2, 2, 5, 2),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=32,
            stem_bias=True,
            head_hidden_size=256,
        ),
        wsfvit_t_224=wsfVitCfg(
            embed_dim=(64, 128, 256, 512),
            depths=(1, 2, 3, 1),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=32,
            stem_bias=True,
            head_hidden_size=512,
        ),
        wsfvit_t_384=wsfVitCfg(
            embed_dim=(64, 128, 256, 512),
            depths=(1, 2, 3, 1),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=32,
            stem_bias=True,
            head_hidden_size=512,
            transformer_cfg=wsfVitTransformerCfg(
                window_size=(12, 12),
                grid_size=(12, 12),
            )
        ),
        wsfvit_s_224=wsfVitCfg(
            embed_dim=(64, 128, 256, 512),
            num_heads=(4, 8, 16, 32),
            depths=(2, 2, 5, 2),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=64,
            stem_bias=True,
            head_hidden_size=512,
        ),
        wsfvit_s_384=wsfVitCfg(
            embed_dim=(64, 128, 256, 512),
            num_heads=(4, 8, 16, 32),
            depths=(2, 2, 5, 2),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=64,
            stem_bias=True,
            head_hidden_size=512,
            transformer_cfg=wsfVitTransformerCfg(
                window_size=(12, 12),
                grid_size=(12, 12),
            )
        ),
        wsfvit_m_224=wsfVitCfg(
            embed_dim=(80, 160, 320, 640),
            num_heads=(4, 8, 16, 32),
            depths=(2, 2, 6, 2),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=80,
            stem_bias=True,
            head_hidden_size=640,
            transformer_cfg=wsfVitTransformerCfg(
                dim_head = 40
            ),
        ),
        wsfvit_m_384=wsfVitCfg(
            embed_dim=(80, 160, 320, 640),
            num_heads=(4, 8, 16, 32),
            depths=(2, 2, 6, 2),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=80,
            stem_bias=True,
            head_hidden_size=640,
            transformer_cfg=wsfVitTransformerCfg(
                dim_head = 40,
                window_size=(12, 12),
                grid_size=(12, 12),
            ),
        ),
        wsfvit_b_224=wsfVitCfg(
            embed_dim=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=64,
            stem_bias=True,
            head_hidden_size=768,
            transformer_cfg=wsfVitTransformerCfg(
                dim_head = 32,
            ),
        ),
        wsfvit_b_256=wsfVitCfg(
            embed_dim=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=64,
            stem_bias=True,
            head_hidden_size=768,
            transformer_cfg=wsfVitTransformerCfg(
                dim_head = 32,
                window_size=(8, 8),
                grid_size=(8, 8),
            ),
        ),
        wsfvit_b_384=wsfVitCfg(
            embed_dim=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
            mlp_ratio=4.,
            block_type=('M',) * 4,
            stem_width=64,
            stem_bias=True,
            head_hidden_size=768,
            transformer_cfg=wsfVitTransformerCfg(
                dim_head = 32,
                window_size=(12, 12),
                grid_size=(12, 12),
            ),
        ),
    )


if classification:
    @register_model
    def wsfvit_p1_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_p1_224', 'wsfvit_p1_224', pretrained=pretrained, **kwargs)
    
    @register_model
    def wsfvit_p2_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_p2_224', 'wsfvit_p2_224', pretrained=pretrained, **kwargs)
    
    @register_model
    def wsfvit_n_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_n_224', 'wsfvit_n_224', pretrained=pretrained, **kwargs)
    
    @register_model
    def wsfvit_t_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_t_224', 'wsfvit_t_224', pretrained=pretrained, **kwargs)
    @register_model
    def wsfvit_t_384(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_t_384', 'wsfvit_t_384', pretrained=pretrained, **kwargs)

    @register_model
    def wsfvit_s_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_s_224', 'wsfvit_s_224', pretrained=pretrained, **kwargs)
    @register_model
    def wsfvit_s_384(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_s_384', 'wsfvit_s_384', pretrained=pretrained, **kwargs)

    @register_model
    def wsfvit_m_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_m_224', 'wsfvit_m_224', pretrained=pretrained, **kwargs)
    @register_model
    def wsfvit_m_384(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_m_384', 'wsfvit_m_384', pretrained=pretrained, **kwargs)

    @register_model
    def wsfvit_b_224(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_b_224', 'wsfvit_b_224', pretrained=pretrained, **kwargs)
    @register_model
    def wsfvit_b_256(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_b_256', 'wsfvit_b_256', pretrained=pretrained, **kwargs)
    @register_model
    def wsfvit_b_384(pretrained=False, **kwargs) -> wsfVit:
        return _create_wsfvit('wsfvit_b_384', 'wsfvit_b_384', pretrained=pretrained, **kwargs)
    
else:
    if mmpose:
        @MODELS.register_module()
        class wsfVit_feat(wsfVit):
            def __init__(self, **kwargs):
                super().__init__(fork_feat=True, **kwargs)
    elif mmdet:
        @det_BACKBONES.register_module()
        class wsfVit_feat(wsfVit):
            def __init__(self, **kwargs):
                super().__init__(fork_feat=True, **kwargs)
    else:
        pass
