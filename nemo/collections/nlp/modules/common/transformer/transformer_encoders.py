# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.omegaconf import MISSING

from nemo.collections.common.parts import form_attention_mask
from nemo.collections.nlp.modules.common.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF, CompositionalAttention, NonlinearMultiHeadAttention, NIAttention, NIPositionWiseFF

__all__ = ["TransformerEncoder"]

class TypeInference(nn.Module):
    def __init__(self, hidden_dim, num_rules, dim):
        super().__init__()

        self.linear = nn.Linear(hidden_dim, dim)
        self.codes = nn.Parameter(torch.randn(1, dim, num_rules))

    def forward(self, x):
        type = self.linear(x) # (bsz, n_tokens, dim)
        score = torch.matmul(type, self.codes).transpose(-2, -1) # (bsz, num_rules, n_tokens)
        score = F.softmax(score, dim=1)
        return score.unsqueeze(-1)

def make_attention(attention_type, hidden_size,
                   num_attention_heads, num_attention_rules,
                   qk_dim, attn_score_dropout,
                   attn_layer_dropout, embedding):
    if attention_type == "MultiHead":
        return MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
    elif attention_type == "Compositional":
        return CompositionalAttention(
            hidden_size, num_attention_heads, num_attention_rules, qk_dim, attn_score_dropout, attn_layer_dropout,
            embedding
        )
    elif attention_type == "NonlinearMultiHeadAttention":
        return NonlinearMultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
    elif attention_type == 'NI' or attention_type == 'NI-Full':
        return NIAttention(
            hidden_size, num_attention_heads, num_attention_rules, qk_dim, attn_score_dropout, attn_layer_dropout
        )

class TransformerEncoderBlock(nn.Module):
    """
    Building block of Transformer encoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        attention_type: str = "MultiHead",
        num_attention_heads: int = 1,
        num_attention_rules: int = 0,
        qk_dim: int = 32,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        embedding: bool = False
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attention_type = attention_type

        self.first_sub_layer = make_attention(attention_type, hidden_size, num_attention_heads, num_attention_rules, qk_dim, attn_score_dropout, attn_layer_dropout, embedding)

        if attention_type == 'NI' or attention_type == 'NI-Full':
            self.codes = nn.Parameter(torch.randn(1, num_attention_rules, 1, qk_dim))
            self.t = TypeInference(hidden_size, num_attention_rules, qk_dim)

        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        if attention_type == 'NI-Full':
            self.second_sub_layer = NIPositionWiseFF(hidden_size, inner_size, qk_dim, ffn_dropout, hidden_act)
        else:
            self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

        if self.pre_ln:
            if attention_type == 'NI-Full':
                self.forward_call = self.forward_preln_ni_full
            elif attention_type == 'NI':
                self.forward_call = self.forward_preln_ni
            else:
                self.forward_call = self.forward_preln
        else:
            if attention_type == 'NI-Full':
                self.forward_call = self.forward_postln_ni_full
            elif attention_type == 'NI':
                self.forward_call = self.forward_postln_ni
            else:
                self.forward_call = self.forward_postln

    def forward_preln_ni_full(self, encoder_query, encoder_mask, encoder_keys):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = encoder_query
        comp_score = self.t(residual)
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output = self.first_sub_layer(encoder_query.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_mask, self.codes, None)
        self_attn_output += residual.unsqueeze(1)

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output, self.codes)
        output_states += residual

        output_states = (output_states * comp_score).sum(dim=1)
        return output_states

    def forward_preln_ni(self, encoder_query, encoder_mask, encoder_keys):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = encoder_query
        comp_score = self.t(residual)
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output = self.first_sub_layer(encoder_query.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_mask, self.codes, comp_score)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual

        return output_states

    def forward_postln_ni_full(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        comp_score = self.t(encoder_query)

        self_attn_output = self.first_sub_layer(encoder_query.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_mask, self.codes, None)
        self_attn_output += encoder_query.unsqueeze(1)
        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output, self.codes)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)

        output_states = (output_states * comp_score).sum(dim=1)
        return output_states

    def forward_postln_ni(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        comp_score = self.t(encoder_query)

        self_attn_output = self.first_sub_layer(encoder_query.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_keys.unsqueeze(1), encoder_mask, self.codes, comp_score)
        self_attn_output += encoder_query
        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)

        return output_states

    def forward_preln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = encoder_query
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual

        return output_states

    def forward_postln(self, encoder_query, encoder_mask, encoder_keys, comp_score=None):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query
        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output
        output_states = self.layer_norm_2(output_states)

        return output_states

    def forward(self, encoder_query, encoder_mask, encoder_keys):
        return self.forward_call(encoder_query, encoder_mask, encoder_keys)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        mask_future: bool = False,
        attention_type: str = "MultiHead",
        num_attention_heads: int = 1,
        num_attention_rules: int = 0,
        qk_dim: int = 32,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        embedding: bool = False
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        layer = TransformerEncoderBlock(
            hidden_size,
            inner_size,
            attention_type,
            num_attention_heads,
            num_attention_rules,
            qk_dim,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            hidden_act,
            pre_ln,
            embedding
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diag = 0 if mask_future else None

    def _get_memory_states(self, encoder_states, encoder_mems_list=None, i=0):
        if encoder_mems_list is not None:
            memory_states = torch.cat((encoder_mems_list[i], encoder_states), dim=1)
        else:
            memory_states = encoder_states
        return memory_states

    def forward(self, encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            encoder_mems_list: list of the cached encoder hidden states
                for fast autoregressive generation which will be used instead
                of encoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """

        encoder_attn_mask = form_attention_mask(encoder_mask, self.diag)

        memory_states = self._get_memory_states(encoder_states, encoder_mems_list, 0)
        cached_mems_list = [memory_states]

        for i, layer in enumerate(self.layers):
            encoder_states = layer(encoder_states, encoder_attn_mask, memory_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if self.final_layer_norm is not None:
            encoder_states = self.final_layer_norm(encoder_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]
