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
from typing import Tuple, Union, Sequence
# coding=utf-8
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import os
import copy
import logging

from functools import reduce, lru_cache
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
#from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_

#from .neural_network import SegmentationNetwork

logger = logging.getLogger(__name__)
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
#from vtunet.network_architecture.neural_network import SegmentationNetwork
import torch
import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import numpy as np
from monai.utils import optional_import
import torch.nn.functional as F
from monai.networks.layers import Conv

einops, _ = optional_import("einops")
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option
import math
#from .discriminator import PatchGAN
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


# np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)

def get_position_embedding_1D(grid_size, hidden_size):
    H, W, S = grid_size
    PE = np.zeros((H, W, S, hidden_size))
    for i in range(H):
        for j in range(W):
            for k in range(S):
                pos = [(H ** 2 * i + W * j + k) / np.power(10000, 2 * (hid // 2) / hidden_size) for hid in
                       range(hidden_size)]

                PE[i, j, k] = np.asarray(pos)
    PE[:, :, :, 0::2] = np.sin(PE[:, :, :, 0::2])
    PE[:, :, :, 1::2] = np.cos(PE[:, :, :, 1::2])
    PE = einops.rearrange(PE, "h w s dim->(w s h) dim")
    PE = torch.FloatTensor(PE).unsqueeze(0)
    PE.requires_grad = False
    return PE


def get_position_embedding_1D_Sym(grid_size, hidden_size):
    H, W, S = grid_size
    PE = np.zeros((H, W, S, hidden_size))
    for i in range(H):
        for j in range(W):
            for k in range(S):
                pos = [(H ** 2 * i + W * j - np.abs(S / 2 - k) + S / 2) / np.power(10000, 2 * (hid // 2) / hidden_size)
                       for hid in
                       range(hidden_size)]

                PE[i, j, k] = np.asarray(pos)
    PE[:, :, :, 0::2] = np.sin(PE[:, :, :, 0::2])
    PE[:, :, :, 1::2] = np.cos(PE[:, :, :, 1::2])
    PE = einops.rearrange(PE, "h w s dim->(w s h) dim")
    PE = torch.FloatTensor(PE).unsqueeze(0)
    PE.requires_grad = False
    return PE


class PatchEmbeddingBlock_vit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int,
            num_heads: int,
            pos_embed: str,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            use_learnable_pos_emb: bool = False,
            symmetry: int = 1,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super(PatchEmbeddingBlock_vit, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = in_channels * np.prod(patch_size)
        self.sym = symmetry
        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)

            # print("Using perceptron HSW")
            # chars = (("w", "p2"), ("d", "p3"), ("h", "p1"))[:spatial_dims]
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i + 1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len),
                nn.Linear(self.patch_dim, hidden_size),
            )
        # learnable
        if use_learnable_pos_emb:
            print("Using Learnable PE")
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
            self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        else:
            if symmetry == 3:  # 3D SPE
                print("Using SYM 3D PE")
                self.position_embeddings = build_3d_sincos_position_sym_embedding(
                    [im_d // p_d for im_d, p_d in zip(img_size, patch_size)], hidden_size)
            elif symmetry == 1:  # 1D SPE
                print("Using SYM 1D PE")
                self.position_embeddings = build_sincos_position_embedding(size=int(self.n_patches / 2),
                                                                           embed_dim=hidden_size,
                                                                           symm=int((img_size[2] // patch_size[2]) / 2))
            elif symmetry == 0:  # 1D PE
                print("Using Cosine 1D PE")
                self.position_embeddings = get_sinusoid_encoding_table(int(self.n_patches), hidden_size)
            elif symmetry == 4:  # 3D PE
                print("Using Cosine 3D PE")
                self.position_embeddings = build_3d_sincos_position_embedding(
                    [im_d // p_d for im_d, p_d in zip(img_size, patch_size)], hidden_size)
            elif symmetry == 5:  # 1D PE hws
                print("Using Cosine 1D PE HWS")
                self.position_embeddings = get_position_embedding_1D(
                    [im_d // p_d for im_d, p_d in zip(img_size, patch_size)], hidden_size)
            elif symmetry == 6:  # 1D SPE hws
                print("Using Cosine 1D SPE HWS")
                self.position_embeddings = get_position_embedding_1D_Sym(
                    [im_d // p_d for im_d, p_d in zip(img_size, patch_size)], hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        if self.sym == 0:
            embeddings = x + self.position_embeddings.expand(x.size(0), -1, -1).type_as(x).to(x.device).clone().detach()
        else:
            embeddings = x + self.position_embeddings.type_as(x).to(x.device)
        embeddings = self.dropout(embeddings)
        return embeddings


def recover_symm(pos_embed, symm):
    pos_embed_left = pos_embed[0, :, :]
    pos_embed_right = pos_embed[1, :, :]
    pos_embed_right = einops.rearrange(pos_embed_right, "(h p1) d->h p1 d", p1=symm)
    pos_embed_right = torch.flip(pos_embed_right, dims=[1])
    pos_embed_left = einops.rearrange(pos_embed_left, "(h p1) d->h p1 d", p1=symm)
    pos_embed = torch.hstack([pos_embed_left, pos_embed_right])
    pos_embed = einops.rearrange(pos_embed, "h w d->(h w) d")
    return pos_embed


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def build_sincos_position_embedding(size, embed_dim, symm, temperature=100000.):
    D = size

    def get_position_angle_vec(position):
        return [position / np.power(temperature, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(D)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    pos_embed = torch.FloatTensor(sinusoid_table)
    pos_embed = torch.stack([pos_embed, pos_embed], dim=0)
    pos_embed = recover_symm(pos_embed, symm)
    pos_embed.requires_grad = False
    return pos_embed


def build_3d_sincos_position_embedding(grid_size, embed_dim, temperature=10000.):
    h, w, s = grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_s = torch.arange(s, dtype=torch.float32)
    grid_w, grid_h, grid_s = torch.meshgrid(grid_w, grid_h, grid_s)
    assert embed_dim % 6 == 0
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_s = torch.einsum('m,d->md', [grid_s.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_s), torch.cos(out_s), torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
        dim=1)[None, :, :]
    pos_emb.requires_grad = False
    # pos_emb = einops.rearrange(pos_emb, "(h w s) p-> h w s p", w=w, s=s)
    return pos_emb


def build_3d_sincos_position_sym_embedding(grid_size, embed_dim, temperature=10000.):
    h, w, s = grid_size
    grid_s = torch.cat(
        [torch.arange(s / 2, dtype=torch.float32), torch.arange(s / 2, dtype=torch.float32).flip(dims=[0])], dim=0)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_w, grid_h, grid_s = torch.meshgrid(grid_w, grid_h, grid_s)
    assert embed_dim % 6 == 0
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_s = torch.einsum('m,d->md', [grid_s.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_s), torch.cos(out_s), torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
        dim=1)[None, :, :]
    pos_emb.requires_grad = False
    # pos_emb = einops.rearrange(pos_emb, "(h w s) p-> h w s p", w=w, s=s)
    return pos_emb


class UnetrPrDownBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            downsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            conv_block: bool = False,
            res_block: bool = False,
    ) -> None:
        super().__init__()
        downsample_stride = downsample_kernel_size
        self.down_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=downsample_kernel_size,
            stride=downsample_stride,
            conv_only=True,
            is_transposed=False,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=downsample_kernel_size,
                                stride=downsample_stride,
                                conv_only=True,
                                is_transposed=False,
                            ),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )

    def forward(self, x):
        x = self.down_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class SWABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            window_size: int = 32,
            qkv_bias: bool = False,
            pretrained_window_size: list = [0, 0, 0]
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super(SWABlock, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale_old = self.head_dim ** -0.5
        self.scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.window_size = window_size
        
        # get relative_coords_table
        relative_coords_d = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_d,relative_coords_h,
                            relative_coords_w])).permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)
        

# ===============================old==========================================
#         self.relative_position_bias_table_old = nn.Parameter(
#             torch.zeros((2 * window_size - 1), num_heads))
# =============================================================================
        
        
        coords_d = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_d]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
#        trunc_normal_(self.relative_position_bias_table_old, std=.02)
        
        self.softmax = nn.Softmax(dim=-1)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(hidden_size))
            self.v_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.q_bias = None
            self.v_bias = None
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(3, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        
        

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        
        # ===========================old======================================
        #q, k, v = einops.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        #att_mat_old = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        
        # =============================New=====================================
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
# =======================================================================
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        
        # cosine attention
        att_mat = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.scale.device)).exp()
        att_mat = att_mat * logit_scale
      
      
        ###     New       ###########################################3
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        att_mat = att_mat + relative_position_bias.unsqueeze(0) # B_, nH, N, N
        
        ###     Old       ###########################################3
# ==================================old=======================================
#         relative_position_bias_old = self.relative_position_bias_table_old[self.relative_position_index.view(-1)].view(
#             self.window_size, self.window_size, -1)  # W,nH
#         relative_position_bias_old = relative_position_bias_old.permute(2, 0, 1).contiguous()
#         att_mat_old = att_mat_old + relative_position_bias_old.unsqueeze(0)
# =============================================================================

        if mask is not None:
            nW = mask.shape[0]
            att_mat = att_mat.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            att_mat = att_mat.view(-1, self.num_heads, N, N)
            att_mat = self.softmax(att_mat)
        else:
            att_mat = self.softmax(att_mat)

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


def window_partition(x, window_size):
    B, D, C = x.shape
    x = x.view(B, D // window_size, window_size, C)
    windows = x.view(-1, window_size, C).contiguous()
    return windows


class SwinTransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            dim,
            num_heads: int,
            input_resolution: int,
            dropout_rate: float = 0.0,
            window_size: int = 32,
            shift_size: int = 0,
            mlp_ratio=4.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()
        self.dim=dim
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        #if hidden_size % num_heads != 0:
            #raise ValueError("hidden_size should be divisible by num_heads.")
        self.mlp_ratio=mlp_ratio
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_hidden_dim, dropout_rate)
        
        #self.mlp = MLPBlock(hidden_size, dim, dropout_rate)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SWABlock(dim, num_heads, dropout_rate, window_size)
        self.norm2 = nn.LayerNorm(dim)

        self.pad = (self.window_size - input_resolution[0] % self.window_size) % self.window_size
        if self.shift_size > 0:
            img_mask = torch.zeros((1, self.input_resolution[0], 1))
            img_mask = F.pad(img_mask, (0, 0, 0, self.pad, 0, 0))
            d_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for d in d_slices:
                img_mask[:, d, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape

        shortcut = x
        

        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, self.pad, 0, 0))

        _, Lp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1,))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = attn_windows.view(B, Lp, C)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1,))
        else:
            x = shifted_x

        if self.pad > 0:
            x = x[:, :L, :].contiguous()
        
        x = self.norm1(x.view(B, L, C))

        x = shortcut + x

        x = x + self.norm2(self.mlp(x))
        return x

class SwinTransformerBlock_2(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            dim,
            num_heads: int,
            input_resolution: int,
            dropout_rate: float = 0.0,
            window_size: int = 32,
            shift_size: int = 0,
            mlp_ratio=4.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()
        self.dim=dim
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        #if hidden_size % num_heads != 0:
            #raise ValueError("hidden_size should be divisible by num_heads.")
        self.mlp_ratio=mlp_ratio
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_hidden_dim, dropout_rate)
        
        #self.mlp = MLPBlock(hidden_size, dim, dropout_rate)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SWABlock(dim, num_heads, dropout_rate, window_size)
        self.norm2 = nn.LayerNorm(dim)

        self.pad = (self.window_size - input_resolution[0] % self.window_size) % self.window_size
        if self.shift_size > 0:
            img_mask = torch.zeros((1, self.input_resolution[0], 1))
            img_mask = F.pad(img_mask, (0, 0, 0, self.pad, 0, 0))
            d_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for d in d_slices:
                img_mask[:, d, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_up,skip):
        B, L, C = x.shape

        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        # skip = skip.view(B, S, H, W, C)
        # x_up = x_up.view(B, S, H, W, C)
        # x = x.view(B, S, H, W, C)
        

        if self.pad > 0:
            x_up = F.pad(x, (0, 0, 0, self.pad, 0, 0))
            skip = F.pad(x, (0, 0, 0, self.pad, 0, 0))

        _, Lp, _ = skip.shape

        if self.shift_size > 0:
            shifted_x_up = torch.roll(x_up, shifts=(-self.shift_size), dims=(1,))
            shifted_skip = torch.roll(skip, shifts=(-self.shift_size), dims=(1,))
        else:
            shifted_x_up = x_up
            shifted_skip=skip
        x_windows = window_partition(shifted_x_up, self.window_size)
        y_windows = window_partition(shifted_skip, self.window_size)
        x_windows=1/2*(x_windows+y_windows)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = attn_windows.view(B, Lp, C)
        if self.shift_size > 0:
            x = torch.roll(shifted_x_up, shifts=(self.shift_size), dims=(1,))
        else:
            x = shifted_x_up

        if self.pad > 0:
            x = x[:, :L, :].contiguous()
        
        x = self.norm1(x.view(B, L, C))

        x = shortcut + x

        x = x + self.norm2(self.mlp(x))
        return x

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
        return



# def window_partition_old(x, window_size):
  
#     B, S, H, W, C = x.shape
#     x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
#     return windows


# def window_reverse(windows, window_size, S, H, W):
   
#     B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
#     x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
#     return x



# class SwinTransformerBlock_kv(nn.Module):


#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention_kv(
#                 dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
#         #self.window_size=to_3tuple(self.window_size)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       
#     def forward(self, x, mask_matrix,skip=None,x_up=None):
    
#         B, L, C = x.shape
#         S, H, W = self.input_resolution
 
#         assert L == S * H * W, "input feature has wrong size"
        
#         shortcut = x
#         skip = self.norm1(skip)
#         x_up = self.norm1(x_up)

#         skip = skip.view(B, S, H, W, C)
#         x_up = x_up.view(B, S, H, W, C)
#         x = x.view(B, S, H, W, C)
#         # pad feature maps to multiples of window size
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         pad_g = (self.window_size - S % self.window_size) % self.window_size

#         skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
#         x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
#         _, Sp, Hp, Wp, _ = skip.shape

       
        
#         # cyclic shift
#         if self.shift_size > 0:
#             skip = torch.roll(skip, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
#             x_up = torch.roll(x_up, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
#             attn_mask = mask_matrix
#         else:
#             skip = skip
#             x_up=x_up
#             attn_mask = None
#         # partition windows
#         skip = window_partition(skip, self.window_size) 
#         skip = skip.view(-1, self.window_size * self.window_size * self.window_size,
#                                    C)  
#         x_up = window_partition(x_up, self.window_size) 
#         x_up = x_up.view(-1, self.window_size * self.window_size * self.window_size,
#                                    C)  
#         attn_windows=self.attn(skip,x_up,mask=attn_mask,pos_embed=None)

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
#         else:
#             x = shifted_x

#         if pad_r > 0 or pad_b > 0 or pad_g > 0:
#             x = x[:, :S, :H, :W, :].contiguous()

#         x = x.view(B, S * H * W, C)

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x
        
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

# class SwinTransformerBlock(nn.Module):
    
#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
   
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)

#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
        
#         self.attn = WindowAttention(
#             dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
       
            

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
       
#     def forward(self, x, mask_matrix):

#         B, L, C = x.shape

#         S, H, W = self.input_resolution
   
#         assert L == S * H * W, "input feature has wrong size"
        
        
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, S, H, W, C)

#         # pad feature maps to multiples of window size
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         pad_g = (self.window_size - S % self.window_size) % self.window_size

#         x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
#         _, Sp, Hp, Wp, _ = x.shape

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
#             attn_mask = mask_matrix
#         else:
#             shifted_x = x
#             attn_mask = None
       
#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
#                                    C)  

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=attn_mask,pos_embed=None)  

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp) 

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
#         else:
#             x = shifted_x

#         if pad_r > 0 or pad_b > 0 or pad_g > 0:
#             x = x[:, :S, :H, :W, :].contiguous()

#         x = x.view(B, S * H * W, C)

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x


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

       
        
        x = self.norm(x)
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
                 downsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
       
        # build blocks
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, 
                num_heads=num_heads, 
                input_resolution=input_resolution,
                dropout_rate=0,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
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
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W

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
        self.blocks.append(
                SwinTransformerBlock_2(
                        dim, 
                        num_heads, 
                        input_resolution=input_resolution,
                        dropout_rate=0,
                        window_size=window_size, shift_size=0 ) )
        
        for i in range(depth-1):
            self.blocks.append(
                SwinTransformerBlock(
                        dim, 
                        num_heads, 
                        input_resolution=input_resolution,
                        dropout_rate=0,
                        window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) )
                        
        

        
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, S, H, W):
        
      
        x_up = self.Upsample(x, S, H, W)
       
        x = x_up + skip
        S, H, W = S * 2, H * 2, W * 2
        # # calculate attention mask for SW-MSA
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

        # mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        # mask_windows = mask_windows.view(-1,
        #                                  self.window_size * self.window_size * self.window_size)  # 3d��3��winds�˻�����Ŀ�Ǻܴ�ģ�����winds����̫��
        # attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        x=self.blocks[0](x,skip=skip,x_up=x_up)
        for i in range(self.depth-1):
            x = self.blocks[i+1](x)
        
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
        x = self.proj1(x)  # B C Ws Wh Ww
        x = self.proj2(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

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
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def forward(self, x):
        """Forward function."""
        
        x = self.patch_embed(x)
        down=[]
       
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)
        
      
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
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
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x,skips):
            
        outs=[]
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        for index,i in enumerate(skips):
             i = i.flatten(2).transpose(1, 2).contiguous()
             skips[index]=i
        x = self.pos_drop(x)
            
        for i in range(self.num_layers)[::-1]:
            
            layer = self.layers[i]
            
            x, S, H, W,  = layer(x,skips[i], S, H, W)
            out = x.view(-1, S, H, W, self.num_features[i])
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
                feature_size = 24,
                num_heads=[6, 12, 24, 48],
                patch_size=[2,4,4],
                window_size=[4,4,8,4],
                dropout_rate = 0.0,
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
        
        
        
   

   
