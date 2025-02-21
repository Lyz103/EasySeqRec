# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 10:57
# @Author  : Hui Wang

import torch
import torch.nn as nn
from modules import *
from torch.nn.init import xavier_uniform_, xavier_normal_


class SRModel(nn.Module):
    def __init__(self, args):
        super(SRModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.args = args
        self.apply(self.init_weights)
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def finetune(self, input_ids):
        pass


class SASRec(SRModel):
    def __init__(self, args):
        super().__init__(args)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)


    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    
    # Fine tune
    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

class Linrec(SRModel):
    def __init__(self, args):
        super().__init__(args)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = LinrecTransformerEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    
    # Fine tune
    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output


class FMLPRecModel(SRModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FMLPEncoder(args)
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    # same as SASRec
    def finetune(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Mamba4Rec(SRModel):
    def __init__(self, args):
        super().__init__(args)

        self.hidden_size = args.hidden_size
        self.dropout_prob = args.hidden_dropout_prob
        # Hyperparameters for Mamba block
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.item_encoder = MambaLayers(args)
        self.apply(self.init_weights)



    def finetune(self, input_ids):
        item_emb = self.item_embeddings(input_ids)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        
        item_encoded_layers = self.item_encoder(item_emb, output_all_encoded_layers=True)
        seq_output = item_encoded_layers[-1]
        return seq_output


class GRU4Rec(SRModel):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__(args)

        # load parameters info
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_hidden_layers
        self.dropout_prob = args.hidden_dropout_prob
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def finetune(self, item_seq):
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        seq_output = self.dense(gru_output)
        return seq_output


class TTT4Rec(SRModel):
    def __init__(self, args):
        super().__init__(args)

        self.hidden_size = args.hidden_size
        self.dropout_prob = args.hidden_dropout_prob
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.item_encoder = TTTLayers(args)
        self.apply(self.init_weights)

    def finetune(self, input_ids):
        item_emb = self.item_embeddings(input_ids)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        
        item_encoded_layers = self.item_encoder(item_emb, output_all_encoded_layers=True)
        seq_output = item_encoded_layers[-1]
        return seq_output
    
class FilterTTT4Rec(SRModel):
    def __init__(self, args):
        super().__init__(args)

        self.hidden_size = args.hidden_size
        self.dropout_prob = args.hidden_dropout_prob
        self.LowPassFilterLayer = LowPassFilterLayer(args)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.item_encoder = TTTLayers(args)
        self.apply(self.init_weights)



    def finetune(self, input_ids):
        item_emb = self.item_embeddings(input_ids)
        # item_emb = self.dropout(item_emb)
        # item_emb = self.LayerNorm(item_emb)
        FilterLayer_output = self.LowPassFilterLayer(item_emb)
        item_encoded_layers = self.item_encoder(FilterLayer_output, output_all_encoded_layers=True)
        seq_output = item_encoded_layers[-1]
        return seq_output


class Narm(SRModel):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__(args)
        # load parameters info
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.n_layers = args.num_hidden_layers
        self.dropout_probs = [args.hidden_dropout_prob / 2, args.hidden_dropout_prob]


        # define layers and loss
        self.item_embedding = nn.Embedding(
            args.item_size, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)

        self.apply(self._init_weights)
    def finetune(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = gru_out[:, -1, :]
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)
        return seq_output

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

from recbole.model.layers import LightTransformerEncoder

class LightSANs(SRModel):
    def __init__(self, args):
        super().__init__(args)

        # load parameters info
        self.n_layers = args.num_hidden_layers
        self.n_heads = args.num_attention_heads
        self.k_interests = 5
        self.hidden_size = args.hidden_size  # same as embedding_size
        self.inner_size = args.hidden_size * 4  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.hidden_dropout_prob
        self.hidden_act = args.hidden_act
        self.initializer_range = args.initializer_range
        self.n_items = args.item_size

        self.seq_len = args.max_seq_length
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.seq_len, self.hidden_size)
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=1e-12,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        # parameters initialization
        self.apply(self.init_weights)

    def embedding_layer(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        return item_emb, position_embedding

    def finetune(self, item_seq):
        item_emb, position_embedding = self.embedding_layer(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)

        trm_output = self.trm_encoder(
            item_emb, position_embedding, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        return output  # [B ,L, H]