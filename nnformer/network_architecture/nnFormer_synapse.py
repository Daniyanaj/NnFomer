from einops import rearrange
from copy import deepcopy
from nnformer.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from functools import partial

import torch
import torch.utils.checkpoint as checkpoint
#pipimport mmsegmentation
#from mmsegmentation.models.builder import BACKBONES
#from mmsegmentation.utils import get_root_logger
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


NORM_EPS = 1e-5

import math
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv3d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BatchNorm3d', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv3d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp00(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN3d', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv3d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        #self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv3d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None,
        norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv3d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv3d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv3d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class TokenPyramidModule(nn.Module):
    def __init__(
        self,
        cfgs,
        out_indices,
        inp_channel=16,
        activation=nn.ReLU,
        norm_cfg=dict(type='BN', requires_grad=True),
        width_mult=1.):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv3d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
            activation()
        )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg, activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BatchNorm3d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads//2  # num_head key_dim
        self.d = int(attn_ratio * key_dim//16)
        self.dh = int(attn_ratio * key_dim//16) * num_heads

        self.attn_ratio = attn_ratio

        self.to_q = Conv3d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv3d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv3d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.p=192

        self.proj = torch.nn.Sequential(activation(), Conv3d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        B, N,C= get_shape(x)
        H=W=S= round(N**(1/3))
        x=x.view(B,C,H,W,S)
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim//2, N).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim//2, N)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, N).permute(0, 1, 3, 2)
        a,b,c,d=qq.shape
        qq=qq.reshape(a,b*32,c//32,d)
        a,b,c,d=kk.shape
        kk = kk.reshape(a, b*32,c,d//32)
        a,b,c,d=vv.shape
        vv = vv.reshape(a, b*32, c//32, d)
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)
        a,b,c,d=xx.shape
        # xx=xx.reshape(a,b//32,c*32,d)
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W,S)
        print(xx.shape)
        # proj = torch.nn.Sequential(Conv3d_BN(    self.dh//(32//H), C, bn_weight_init=0, norm_cfg=dict(type='BN3d', requires_grad=True))).cuda
        xx = self.proj(xx)
        a,b,c,d,e=xx.shape
        xx=xx.reshape(B,H*W*S,C)
        return xx

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
  
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
   
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x



class SwinTransformerBlock_kv(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
                dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        #self.window_size=to_3tuple(self.window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       
    def forward(self, x, mask_matrix,skip=None,x_up=None):
    
        B, L, C = x.shape
        S, H, W = self.input_resolution
 
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        x = x.view(B, S, H, W, C)
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = skip.shape

       
        
        # cyclic shift
        if self.shift_size > 0:
            skip = torch.roll(skip, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            x_up = torch.roll(x_up, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            attn_mask = mask_matrix
        else:
            skip = skip
            x_up=x_up
            attn_mask = None
        # partition windows
        skip = window_partition(skip, self.window_size) 
        skip = skip.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  
        x_up = window_partition(x_up, self.window_size) 
        x_up = x_up.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  
        attn_windows=self.attn(skip,x_up,mask=attn_mask,pos_embed=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        
class WindowAttention_kv(nn.Module):
   
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w])) 
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, skip,x_up,pos_embed=None, mask=None):

        B_, N, C = skip.shape
        
        kv = self.kv(skip)
        q = x_up

        kv=kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(B_,N,self.num_heads,C//self.num_heads).permute(0,2,1,3).contiguous()
        k,v = kv[0], kv[1]  
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads)) 

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None,pos_embed=None):

        B_, N, C = x.shape
        
        qkv = self.qkv(x)
        
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x+pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,norm_cfg=dict(type='BN3d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
   
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        self.attn =  Attention(dim, key_dim=dim, num_heads=num_heads, attn_ratio=2., activation=act_layer, norm_cfg= norm_cfg)
       
            

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
       
    def forward(self, x):
        #B,C,S,H,W=x.shape
        #x=x.view(B, S*H*W, C)
        B, L, C = x.shape
        S, H, W = self.input_resolution
   
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        #print(shortcut.shape)

        # FFN
        x = shortcut + self.drop_path(self.attn(x))
        x=self.norm2(x)
        # x=x.view(B,C,S,H,W)
        x=self.mlp(x)
        x = x + self.drop_path(x)
        # a,b,c,d,e=x.shape
        # x=x.view(a,c*d*e,b)


        return x


class PatchMerging(nn.Module):
  

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim,dim*2,kernel_size=3,stride=2,padding=1)
       
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):

        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x=self.reduction(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,2*C)
      
        return x
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
       
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose3d(dim,dim//2,2,2)
    def forward(self, x, S, H, W):
      
        
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

       
        
        #x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,C//2)
       
        return x
class BasicLayer(nn.Module):
   
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 downsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = [2,2,2,2]
        # build blocks
        
        self.blocks = nn.ModuleList([

            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_cfg=norm_cfg)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
      

        # calculate attention mask for SW-MSA
        # Sp = int(np.ceil(S / self.window_size)) * self.window_size
        # Hp = int(np.ceil(H / self.window_size)) * self.window_size
        # Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        # s_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        # h_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        # w_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        # cnt = 0
        # for s in s_slices:
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, s, h, w, :] = cnt
        #             cnt += 1

        # mask_windows = window_partition(img_mask, self.window_size)  
        # mask_windows = mask_windows.view(-1,
        #                                  self.window_size * self.window_size * self.window_size)  
        # attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
          
            x = blk(x)
        #l,m,S,H,W=x.shape
        #x = rearrange(x, "b c h w k-> b (h w k) c")  # b n c      
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H,W

          
        

class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        

        # build blocks
        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=window_size // 2 ,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                        )
        

        
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, S, H, W):
        
      
        x_up = self.Upsample(x, S, H, W)


        x = x_up + skip
        S, H, W = S * 2, H * 2, W * 2
        # calculate attention mask for SW-MSA

        
        #x = self.blocks[0](x, attn_mask,skip=skip,x_up=x_up)
        for i in range(self.depth):
            x = self.blocks[i](x)
        
        return x, S, H, W
        
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x
        
    

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1=[patch_size[0],patch_size[1]//2,patch_size[2]//2]
        stride2=[patch_size[0]//2,patch_size[1]//2,patch_size[2]//2]
        self.proj1 = project(in_chans,embed_dim//2,stride1,1,nn.GELU,nn.LayerNorm,False)
        self.proj2 = project(embed_dim//2,embed_dim,stride2,1,nn.GELU,nn.LayerNorm,True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        #print("x",x.shape)
        x = self.proj1(x)  # B C Ws Wh Ww
        #print("x after proj1",x.shape)
        x = self.proj2(x)  # B C Ws Wh Ww
        #print("x after proj2",x.shape)
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)
        #print("x after norm",x.shape)
        return x



class Encoder(nn.Module):
   
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1  ,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3)
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

       

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
   
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                #dim=int(embed_dim),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None
                )
            self.layers.append(layer)
        #print(len(self.layers))
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        #num_features = [int(embed_dim) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def forward(self, x):
        """Forward function."""
        
        x = self.patch_embed(x)
        #print("x after embed-encod",x.shape)
        down=[]
       
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        
        x = x.flatten(2).transpose(1, 2).contiguous()
        #print("x after flatten",x.shape)    
        x = self.pos_drop(x)
        #print("x after pos",x.shape)    
      
        for i in range(self.num_layers):
            
            layer = self.layers[i]
            
                

            x_out, S, H, W, x,Ws,Wh,Ww= layer(x, Ws, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')

                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
              
            down.append(out)
        return down

   

class Decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2,2,2],
                 num_heads=[24,12,6],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths)-i_layer-1)),
                #dim=int(embed_dim),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1), pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths)-i_layer-1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths)-i_layer-1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
                )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** (len(depths)-i_layer-1 ) )for i in range(self.num_layers)]
    def forward(self,x,skips):
            
        outs=[]
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        for index,i in enumerate(skips):
             i = i.flatten(2).transpose(1, 2).contiguous()
             skips[index]=i
        x = self.pos_drop(x)
        self.num_features=[192,384,768]
        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]

            x, S, H, W, = layer(x, skips[i], S, H, W)
            out= x.view(-1, S, H, W, (self.num_features[i]))
            outs.append(out)
        return outs

        

      
class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.up=nn.ConvTranspose3d(dim,num_class,patch_size,patch_size)
      
    def forward(self,x):
        x=x.permute(0,4,1,2,3).contiguous()
        x=self.up(x)
      
        
        return x    




                                         
class nnFormer(SegmentationNetwork):

    def __init__(self, crop_size=[64,128,128],
                embedding_dim=192,
                input_channels=1, 
                num_classes=14, 
                conv_op=nn.Conv3d, 
                depths=[2,2,2,2],
                num_heads=[4,8,16,32],
                #num_heads=[6, 12, 24, 48],
                patch_size=[2,4,4],
                window_size=[4,4,8,4],
                deep_supervision=True):
      
        super(nnFormer, self).__init__()
        
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes=num_classes
        self.conv_op=conv_op
       
        
        self.upscale_logits_ops = []
     
        
        self.upscale_logits_ops.append(lambda x: x)
        
        embed_dim=embedding_dim
        depths=depths
        num_heads=num_heads
        patch_size=patch_size
        window_size=window_size
        self.model_down=Encoder(pretrain_img_size=crop_size,window_size=window_size,embed_dim=embed_dim,patch_size=patch_size,depths=depths,num_heads=num_heads,in_chans=input_channels)
        self.decoder=Decoder(pretrain_img_size=crop_size,embed_dim=embed_dim,window_size=window_size[::-1][1:],patch_size=patch_size,num_heads=num_heads[::-1][1:],depths=depths[::-1][1:])
        
        self.final=[]
        if self.do_ds:
            
            for i in range(len(depths)-1):
                self.final.append(final_patch_expanding(embed_dim*2**i,num_classes,patch_size=patch_size))

        else:
            self.final.append(final_patch_expanding(embed_dim,num_classes,patch_size=patch_size))
    
        self.final=nn.ModuleList(self.final)
    

    def forward(self, x):
      
            
        seg_outputs=[]
        skips = self.model_down(x)
        neck=skips[-1]
       
        out=self.decoder(neck,skips)
        
       
            
        if self.do_ds:
            for i in range(len(out)):  
                seg_outputs.append(self.final[-(i+1)](out[i]))
        
          
            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]
        
        
        
   

   
