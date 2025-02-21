# -*- coding:utf-8 -*-
# __author__ = 'wanghui'
# __date__ = '2020/03/30 10:53'

import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class LinrecAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self,args):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)   #row-wise
        self.softmax_col = nn.Softmax(dim=-2)   #column-wise
        self.scale = np.sqrt(args.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

#         # Standard Attention
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer)
#         # attention_scores = 0
#         attention_scores = attention_scores / self.sqrt_attention_head_size
#         X=self.softmax(attention_scores)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # [batch_size heads seq_len seq_len] scores
#         # [batch_size 1 1 seq_len]
#         attention_scores = attention_scores + attention_mask
#         # Normalize the attention scores to probabilities.
#         attention_probs = self.softmax(attention_scores)
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.attn_dropout(attention_probs)    
#         context_layer = torch.matmul(attention_probs, value_layer)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)       
        query_norm_inverse = 1/torch.norm(elu_query, dim=3,p=2) #(L2 norm)
        key_norm_inverse = 1/torch.norm(elu_key, dim=2,p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij',elu_query,query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij',elu_key,key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer,torch.matmul(normalized_key_layer,value_layer))/ self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class LinrecTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(self,args):
        super().__init__()
        self.multi_head_attention = LinrecAttention(args)
        self.feed_forward = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LinrecTransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,args
    ):
        super().__init__()
        layer = LinrecTransformerLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = EncoderLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MambaLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_size
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.num_layers = args.num_hidden_layers
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            )
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=self.d_model, inner_size=self.d_model*4, dropout=args.hidden_dropout_prob)
    
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FMLPLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):

        hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class FMLPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        layer = FMLPLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MambaLayers(nn.Module):
    def __init__(self, args):
        super(MambaLayers, self).__init__()
        layer = MambaLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k

def undo_permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually undo the reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float().to(x.device)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # print("---------", device_type)
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    
from transformers.utils import logging
from collections import defaultdict
logger = logging.get_logger(__name__)

class TTTCache:
    """
    TTTCache is a data structure that holds the last hidden states and gradients for the TTT layer.

    Arguments:
        model: TTTModel
        batch_size: int

    Attributes:
        seqlen_offset: int
        mini_batch_size: int
        params_dict: Dict[str, Dict[int, torch.Tensor]]  *_states, *_grad -> # layer_idx -> [batch_size, ...]
        conv_states_dic: Dict[str, Dict[int, torch.Tensor]]  *_states -> # layer_idx -> [batch_size, ...]

    """

    def __init__(self, model, batch_size: int, mini_batch_size: int):
        self.seqlen_offset = 0
        self.mini_batch_size = mini_batch_size

        self.ttt_params_dict = defaultdict(dict)
        self.ttt_param_names = ["W1", "b1"]
        dict_weight = {"W1": model.W1, "b1": model.b1}

        self.conv_states_dic = defaultdict(dict)
        # logger.info(f"Creating cache of size: {batch_size}")
        
        for name in self.ttt_param_names:
            weight = dict_weight[name].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            self.ttt_params_dict[f"{name}_states"] = weight
            # for decoding, we need to store the gradients as well
            self.ttt_params_dict[f"{name}_grad"] = torch.zeros_like(weight)

    def update(self, py_tree, seq_len):
        if seq_len % self.mini_batch_size == 0:
            # copy last mini-batch states, clear gradients
            for name in self.ttt_param_names:
                self.ttt_params_dict[f"{name}_states"].copy_(py_tree[f"{name}_states"])
                self.ttt_params_dict[f"{name}_grad"].zero_()
        elif seq_len < self.mini_batch_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.mini_batch_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.mini_batch_size == 0:
                # copy last mini-batch states, clear gradients
                for name in self.ttt_param_names:
                    self.ttt_params_dict[f"{name}_states"].copy_(py_tree[f"{name}_states"])
                    self.ttt_params_dict[f"{name}_grad"].zero_()
            else:
                # copy gradients for the next update
                for name in self.ttt_param_names:
                    self.ttt_params_dict[f"{name}_grad"].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")

    def ttt_params_to_dict(self):
        return {name: self.ttt_params_dict[name] for name in self.ttt_params_dict}

from torch import nn
import torch

class TTTBase(nn.Module):
    def __init__(self, num_heads, hidden_size, mini_batch_size, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        # print("-----", hidden_size)
        # print("-----", num_heads)
        # print("-----", self.head_dim)


        self.mini_batch_size = mini_batch_size
        self.rope_theta = rope_theta
        
        # token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        # make the scale factor learnable
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))
        
        self._init_qkvo_proj()
        self._init_rope()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()
        
        self.post_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ttt_base_lr = 1.0
        
    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False) 
        
    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.rope_theta,
        )
        
    def _init_ttt_lr_gate(self):
        # [width, 1]
        linear_weight_data = nn.Linear(self.hidden_size, 1, bias=True).weight.data
        # prepending head dim -> [num_heads, width, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.hidden_size, 1, bias=True).bias.data
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data

        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))
    
    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        return XQ, XK, XV
    
    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        # [B, num_heads, num_mini_batch, 1, mini_batch_size] 32 8 2 1 64
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.ttt_base_lr * ttt_lr / self.head_dim

        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [B, num_heads, num_mini_batch, mini_batch_size, 1] [32, 8, 2, 64, 1]
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta
    
    def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        # [B ,num_mini_batch, mini_batch_size, C] 32 2 64 512
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.hidden_size)
        # [B, num_heads, L, head_dim] -> [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        if cache_params is not None:
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        
        # [B, num_heads, num_mini_batch, mini_batch_size, mini_batch_size]
        eta = token_eta * ttt_lr_eta
        # decouple token_coeff and ilr_coeff for decoding
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs
    
    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")
    
    def forward(self, hidden_states, position_ids, cache_params):
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None
        
        XQ, XK, XV = self.get_qkv_projections(hidden_states)
        
        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim] 32, 8, 128, 64
        # print(XQ.shape)
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)
        XQ, XK = permute_qk(XQ, XK)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = undo_permute_qk(XQ, XK)
        
        output_hidden_states = []
        # when input sequence length is not a multiple of mini_batch_size
        # we need to compute them seperately, when computing the reminder,
        # we will need the last_mini_batch_params_dict to continue TTT learning
        # 说人话就是凑整了用对偶的方法求解，没凑整只能一个一个求
        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            output_mod, last_mini_batch_params_dict = self.ttt(
                self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder, _ = self.ttt(
                self.get_ttt_inputs(inputs, reminder_len, cache_params),
                mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_reminder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states
    
def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out

def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z

class TTTLinear(TTTBase):
    def __init__(self, num_heads, hidden_size, mini_batch_size, rope_theta):
        super().__init__(num_heads, hidden_size, mini_batch_size, rope_theta)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        
    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params,
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # in this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict()

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim 64
            W1_init = params_dict["W1_states"].clone()
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"].clone()

            # [B, nh, K,f], K=mini_batch_size  2, 32, 8, 64, 64
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B,nh,K,f]
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                # [B,nh,K,K]
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim] ->
        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] 2, 32, 8, 64, 64
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            0
        )
        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dict, L)

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] ->
        # [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.hidden_size)
        
        return XQW_batch, batch_params_dict


class TTTLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_hidden_layers
        self.ttt = TTTLinear(args.num_TTT_heads, args.hidden_size, args.mini_batch_size, args.rope_theta)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.ffn = FeedForward(d_model=args.hidden_size, inner_size=args.hidden_size*4, dropout=args.hidden_dropout_prob)
        self.mini_batch_size = args.mini_batch_size

    def forward(self, input_tensor):
        B, L = input_tensor.shape[:2]
        cache = TTTCache(self.ttt, B, self.mini_batch_size)
        position_ids = torch.arange(L).unsqueeze(0).repeat(B, 1)
        hidden_states = self.ttt(input_tensor, position_ids, cache)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class TTTLayers(nn.Module):
    def __init__(self, args):
        super().__init__()
        layer = TTTLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.args = args


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = self.LayerNorm(hidden_states)


        return hidden_states

class LowPassFilterLayer(nn.Module):
    def __init__(self, args):
        super(LowPassFilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        # self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        # self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        # self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.args = args

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        # batch, seq_len, hidden = input_tensor.shape
        # normalized_cutoff = seq_len / 2
        # # 计算FFT的输出形状
        # shape = input_tensor.shape
        # fft_shape = (shape[0], shape[1] // 2 + 1, shape[2])
        # x = torch.fft.rfft(input_tensor, dim=1) # 使用onesided=True参数，这意味着torch.rfft只返回正频率部分，这是低通滤波的常见做法
        # # 计算滤波器掩码
        # mask = torch.zeros(fft_shape, device=input_tensor.device)
        # mask[..., :int(normalized_cutoff * shape[1])] = 1  # 低频部分
        # filtered_x = x * mask
        # sequence_emb_fft = torch.fft.irfft(filtered_x, n=seq_len, dim=1, norm='ortho')
        # hidden_states = self.out_dropout(sequence_emb_fft)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        batch, seq_len, hidden = input_tensor.shape
        sample_rate = 20  # 1000和10000,200效果一样，说明频率区间在[0,25]之内，10不行
        # 计算信号的FFT
        signal_fft = torch.fft.rfft(input_tensor, dim=1, norm='ortho') # [batch, seq_len/2+1, hidden]
        # print("signal_fft.shape", signal_fft.shape)
        # 计算信号的频率分量
        freqs = torch.fft.fftfreq(seq_len, 1.0 / sample_rate, device=input_tensor.device)
        # print("freqs.shape", freqs.shape)
        max_freq = sample_rate / 2.0
        
        # 设置截止频率为频率范围的一半
        cutoff_freq = int(max_freq / 2)
        # low_pass_mask = torch.abs(freqs) < cutoff_freq
        # print("low_pass_mask.shape", low_pass_mask.shape)
        # 生成频率掩码
        freq_mask = torch.ones(seq_len // 2 + 1, device=input_tensor.device)
        freq_mask[cutoff_freq:] = 0
        
        # 将频率掩码扩展到与 fft_signal 的形状一致
        extended_freq_mask = freq_mask.unsqueeze(0).unsqueeze(-1).expand(batch, -1, hidden)

        # 将高于截止频率的频率成分置为0
        signal_fft_filtered = signal_fft * extended_freq_mask
        
        # 对滤波后的FFT进行逆变换以获得滤波后的信号
        signal_filtered = torch.fft.irfft(signal_fft_filtered, n=seq_len, dim=1, norm='ortho')
        
        return signal_filtered
        # return sequence_emb_fft

