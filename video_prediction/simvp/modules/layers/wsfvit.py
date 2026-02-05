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


@dataclass
class WSFVitTransformerCfg:
    dim_head: int = 40
    #dim_head: int = 32
    head_first: bool = True  # head ordering in qkv channel dim
    expand_ratio: float = 4.0
    expand_first: bool = True
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio: int = 32
    window_size: Optional[Tuple[int, int]] = (7, 7)
    grid_size: Optional[Tuple[int, int]] = (7, 7)
    no_block_attn: bool = False  # disable window block attention for maxvit (ie only grid)
    use_nchw_attn: bool = False  # for WSFViT variants (not used for CoAt), keep tensors in NCHW order
    init_values: Optional[float] = None
    act_layer: str = 'gelu'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class WSFVitConvCfg:
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
    padding: str = ''
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

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
class WSFVitCfg:
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    num_heads: Tuple[int, ...] = (4, 8, 16, 32)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    mlp_ratio: float = None
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'T', 'T')
    stem_width: Union[int, Tuple[int, int]] = 64
    stem_bias: bool = False
    conv_cfg: WSFVitConvCfg = field(default_factory=WSFVitConvCfg)
    transformer_cfg: WSFVitTransformerCfg = field(default_factory=WSFVitTransformerCfg)
    head_hidden_size: int = None
    weight_init: str = 'vit_eff'

    
class MbConvBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: WSFVitConvCfg = WSFVitConvCfg(),
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
        
        # ============= Expand =============
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1)
        self.norm1 = norm_act_layer(mid_chs)

        # ============= Top =============
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

        # ============= Reverse =============
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=cfg.output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        
        x = self.down(x)

        # =========== Expand ===========
        x = self.conv1_1x1(x)
        x = self.norm1(x)

        # =========== Depth Wise ===========
        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)

        # =========== Reverse ===========
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
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

class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: WSFVitTransformerCfg = WSFVitTransformerCfg(),
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
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = window_reverse(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def window_partition(x, window_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, '')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


def get_rel_pos_cls(cfg: WSFVitTransformerCfg, window_size):
    rel_pos_cls = None
    if cfg.rel_pos_type == 'mlp':
        rel_pos_cls = partial(RelPosMlp, window_size=window_size, hidden_dim=cfg.rel_pos_dim)
    elif cfg.rel_pos_type == 'bias':
        rel_pos_cls = partial(RelPosBias, window_size=window_size)
    elif cfg.rel_pos_type == 'bias_tf':
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
    return rel_pos_cls

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
