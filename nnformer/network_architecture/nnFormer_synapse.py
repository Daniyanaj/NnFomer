from this import d
from einops import rearrange
from copy import deepcopy
from nnformer.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from torch.nn.functional import linear, softmax, dropout 
import math
from torch._C import _infer_size, _add_docstr
from torch.nn import _reduction as _Reduction
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn import grad  # noqa: F401
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
import warnings

#from mmdet3d.ops import flat2window_v2, window2flat_v2

#from ipdb import set_trace
import os
import pickle as pkl


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
Tensor = torch.Tensor

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




def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_cosine_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    tau, 
    tau_min,
    num_heads,
    attn_mask: Optional[Tensor] = None,
    extra_attn: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    Ns = k.shape[1]
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if tau is not None:
        q = nn.functional.normalize(q, dim=2)
        k = nn.functional.normalize(k, dim=2)
        attn = torch.bmm(q, k.transpose(-2, -1))

        if tau.ndim == 4:
            assert tau.size(1) == num_heads and attn.size(-1) == Ns
            attn = attn.reshape(B // num_heads, num_heads, Nt, Ns)
            attn = attn / tau.clamp(min=tau_min)
            attn = attn.reshape(B, Nt, Ns)
        else:
            attn = attn / tau.clamp(min=tau_min)
    else:
        q = q / math.sqrt(E)
        attn = torch.bmm(q, k.transpose(-2, -1))

    if attn_mask is not None:
        attn += attn_mask
    if extra_attn is not None:
        assert extra_attn.shape == attn.shape, f'{extra_attn.shape} v.s. {attn.shape}'
        attn += extra_attn
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def cosine_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    extra_attn: Optional[Tensor] = None,
    tau = None,
    tau_min = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # The following commented lines requires higher version of torch
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if has_torch_function(tens_ops):
    #     raise NotImplementedError('This line should not be visited.')
    #     return handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #     )

    # assert pos embedding

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_cosine_attention(q, k, v, tau, tau_min, num_heads, attn_mask, extra_attn, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class CosineMultiheadAttention(nn.MultiheadAttention):
    '''Inherit from standard multihead attention, call the customized multi_head_forward function in forward.
    '''

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, cosine=True, tau_min=0.01, non_shared_tau=False) -> None:
        super(CosineMultiheadAttention, self).__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.batch_first = batch_first

        self.tau_min = tau_min
        if cosine:
            if non_shared_tau:
                self.tau = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1)) # shared between heads
            else:
                self.tau = torch.nn.Parameter(torch.ones(1, 1, 1)) # shared between heads
        else:
            self.tau = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                extra_attn=None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = cosine_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, extra_attn=extra_attn, tau=self.tau, tau_min=self.tau_min)
        else:
            attn_output, attn_output_weights = cosine_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, extra_attn=extra_attn, tau=self.tau, tau_min=self.tau_min)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info, padding=0):
    '''
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].
    
    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    '''
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}

    for dl in drop_info:

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]

        this_inds = flat2win_inds_dict[dl][0]

        max_tokens = drop_info[dl]['max_tokens']
        num_windows = (this_inds // max_tokens).max().item() + 1
        padding = torch.tensor(padding, dtype=dtype, device=device)
        feat_3d = torch.ones((num_windows * max_tokens, feat_dim), dtype=dtype, device=device) * padding
        # if this_inds.max() >= num_windows * max_tokens:
        #     set_trace()
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict

def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []

    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    # check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        # check_feat[flat_pos] = 0
        # flat_feat_list.append(flat_feat)
    # assert (check_feat == 0).all()
    
    return all_flat_feat
def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)

def flat2window_v2(feat, inds_dict, padding=0):
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict in v2 function'
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info, padding=padding)



class WindowAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout, batch_first=False, layer_id=None):
        super().__init__()
        nhead = num_heads
        d_model=dim

        # from mmdet3d.models.transformer.my_multi_head_attention import MyMultiheadAttention
        # self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.self_attn = CosineMultiheadAttention(d_model, nhead, dropout=dropout)
        self.exe_counter = 0

        self.layer_id = layer_id

    def forward(self, feat_2d, pos_dict, ind_dict, voxel_drop_level, key_padding_dict, drop_info):
        '''
        Args:
        Out:
            shifted_feat_dict: the same type as window_feat_dict
        '''

        out_feat_dict = {}

        feat_3d_dict = flat2window_v2(feat_2d, voxel_drop_level, ind_dict, drop_info)

        for name in feat_3d_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]

            feat_3d = feat_3d_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)

            v = feat_3d

            if pos is not None:
                pos = pos.permute(1, 0, 2)
                assert pos.shape == feat_3d.shape
                q = k = feat_3d + pos
            else:
                q = k = feat_3d

            key_padding_mask = key_padding_dict[name]
            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        results = window2flat_v2(out_feat_dict, ind_dict)
        
        return results



        



class SwinTransformerBlock(nn.Module):


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
        # self.attn = WindowAttention(
        #         dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
        #         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn =  WindowAttention(dim, num_heads, dropout=0, layer_id=None)

        self.mlp_tokens = Mlp(in_features=4096, hidden_features=4096, act_layer=act_layer, drop=drop)
        self.mlp_tokens_1 = Mlp(in_features=512, hidden_features=512, act_layer=act_layer, drop=drop)
        self.mlp_tokens_2 = Mlp(in_features=64, hidden_features=64, act_layer=act_layer, drop=drop)        
        
        #self.window_size=to_3tuple(self.window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       
    def forward(self, x, mask_matrix):
    
        B, L, C = x.shape
        S= H=W = self.input_resolution[0]
   
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
    
        #partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                C)  

        # W-MSA/SW-MSA
        attn_windows = WindowAttention(dim, nhead, dropout, layer_id=layer_id)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp) 

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


        # if L==32768:
        #     S= H= W = 32
            
        #     assert L == S * H * W, "input feature has wrong size"
            
        #     shortcut = x
        #     skip = self.norm1(skip)
        #     x_up = self.norm1(x_up)

        #     skip = skip.view(B, S, H, W, C)
        #     x_up = x_up.view(B, S, H, W, C)
        #     x = x.view(B, S, H, W, C)
        #     # pad feature maps to multiples of window size
        #     pad_r = (self.window_size - W % self.window_size) % self.window_size
        #     pad_b = (self.window_size - H % self.window_size) % self.window_size
        #     pad_g = (self.window_size - S % self.window_size) % self.window_size

        #     skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        #     x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        #     _, Sp, Hp, Wp, _ = skip.shape

       
        
        #     # cyclic shift
        #     if self.shift_size > 0:
        #         skip = torch.roll(skip, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
        #         x_up = torch.roll(x_up, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
        #         attn_mask = mask_matrix
        #     else:
        #         skip = skip
        #         x_up=x_up
        #         attn_mask = None
        #     # partition windows
        #     skip = window_partition(skip, self.window_size) 
        #     skip = skip.view(-1, self.window_size * self.window_size * self.window_size,
        #                             C)  
        #     x_up = window_partition(x_up, self.window_size) 
        #     x_up = x_up.view(-1, self.window_size * self.window_size * self.window_size,
        #                             512)  
        #     attn_windows=self.mlp_tokens_1(x_up)

        #     # merge windows
        #     attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        #     shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)  # B H' W' C

        #     # reverse cyclic shift
        #     if self.shift_size > 0:
        #         x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        #     else:
        #         x = shifted_x

        #     if pad_r > 0 or pad_b > 0 or pad_g > 0:
        #         x = x[:, :S, :H, :W, :].contiguous()

        #     x = x.view(B, S * H * W, C)

        #     # FFN
        #     x = shortcut + self.drop_path(x)
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))

        #     return x

        # elif L==4096:
            
        #     mlp=self.mlp_tokens
        #     shortcut = x
        #     x = self.norm1(x).transpose(1, 2)
        #     x= mlp(x).transpose(1, 2)
        #     x = shortcut + self.drop_path(x)
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     return x

        
        # elif L==512:
            
        #     mlp= self.mlp_tokens_1
        #     shortcut = x
        #     x = self.norm1(x).transpose(1, 2)
        #     x= mlp(x).transpose(1, 2)
        #     x = shortcut + self.drop_path(x)
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     return x

        # else:
            
        #     mlp=self.mlp_tokens_2
        #     shortcut = x
        #     x = self.norm1(x).transpose(1, 2)
        #     x= mlp(x).transpose(1, 2)
        #     x = shortcut + self.drop_path(x)
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     return x    
              
            

        
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

class WindowiAttention(nn.Module):

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
        
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.mlp_tokens = Mlp(in_features=4096, hidden_features=4096, act_layer=act_layer, drop=drop)
        self.mlp_tokens_1 = Mlp(in_features=512, hidden_features=512, act_layer=act_layer, drop=drop)
        self.mlp_tokens_2 = Mlp(in_features=64, hidden_features=64, act_layer=act_layer, drop=drop)
        # self.mlp_tokens_3 = Mlp(in_features=8, hidden_features=8, act_layer=act_layer, drop=drop)
            

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
       
    def forward(self, x, mask_matrix):

        B, L, C = x.shape
        if L==32768:
            S= H=W = 32
   
            assert L == S * H * W, "input feature has wrong size"
            
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, S, H, W, C)

            # pad feature maps to multiples of window size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_g = (self.window_size - S % self.window_size) % self.window_size

            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
            _, Sp, Hp, Wp, _ = x.shape

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None
        
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                    512)  

            # W-MSA/SW-MSA
            attn_windows = self.mlp_tokens_1(x_windows) 

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp) 

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


        elif L==4096:
            
            mlp=self.mlp_tokens
            shortcut = x
            x = self.norm1(x).transpose(1, 2)
            x= mlp(x).transpose(1, 2)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        
        elif L==512:
            
            mlp= self.mlp_tokens_1
            shortcut = x
            x = self.norm1(x).transpose(1, 2)
            x= mlp(x).transpose(1, 2)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        else:
            
            mlp=self.mlp_tokens_2
            shortcut = x
            x = self.norm1(x).transpose(1, 2)
            x= mlp(x).transpose(1, 2)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x    
          

        # FFN
        


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
            SwinTransformerBlock_kv(
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
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
      

        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
          
            x = blk(x, attn_mask)
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
        self.dim=dim
        

        # build blocks
       
        if self.dim==768:
            self.blocks=nn.ModuleList([
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
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                for i in range(depth)])
        else:        
            self.blocks = nn.ModuleList()
            self.blocks.append(
                SwinTransformerBlock_kv(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 ,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                        )
            for i in range(depth-1):
                self.blocks.append(
                    SwinTransformerBlock_kv(
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
                            drop_path=drop_path[i+1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                            )
            

        
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, S, H, W):
        
      
        x_up = self.Upsample(x, S, H, W)
       
        x = x_up + skip
        S, H, W = S * 2, H * 2, W * 2
        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  # 3d��3��winds�˻�����Ŀ�Ǻܴ�ģ�����winds����̫��
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        x = self.blocks[0](x, attn_mask)
        for i in range(self.depth-1):
            x = self.blocks[i+1](x,attn_mask)
            
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
                num_heads=[6, 12, 24, 48],
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
        
        