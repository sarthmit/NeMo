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

__all__ = ["TransformerDecoder"]

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

class TransformerDecoderBlock(nn.Module):
    """
    Building block of Transformer decoder.

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
        self.codes = None

        self.first_sub_layer = make_attention(attention_type, hidden_size, num_attention_heads, num_attention_rules, qk_dim, attn_score_dropout, attn_layer_dropout, embedding)

        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)

        self.second_sub_layer = make_attention(attention_type,
                                              hidden_size,
                                              num_attention_heads,
                                              num_attention_rules,
                                              qk_dim,
                                              attn_score_dropout,
                                              attn_layer_dropout,
                                              embedding
                                              )

        if attention_type == 'NI' or attention_type == 'NI-Full':
            self.codes = nn.Parameter(torch.randn(1, num_attention_rules, 1, qk_dim))
            self.t = TypeInference(hidden_size, num_attention_rules, qk_dim)

        self.layer_norm_3 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)
        if attention_type == 'NI-Full':
            self.third_sub_layer = NIPositionWiseFF(hidden_size, inner_size, qk_dim, ffn_dropout, hidden_act)
        else:
            self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

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

    def forward_preln_ni_full(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = decoder_query
        comp_score = self.t(residual)

        decoder_query = self.layer_norm_1(decoder_query)
        decoder_keys = self.layer_norm_1(decoder_keys)
        self_attn_output = self.first_sub_layer(decoder_query.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_mask, self.codes, None)
        self_attn_output += residual.unsqueeze(1)

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        enc_dec_attn_output = self.second_sub_layer(self_attn_output, encoder_states.unsqueeze(1), encoder_states.unsqueeze(1), encoder_mask, self.codes, None)
        enc_dec_attn_output += residual

        residual = enc_dec_attn_output
        enc_dec_attn_output = self.layer_norm_3(enc_dec_attn_output)
        output_states = self.third_sub_layer(enc_dec_attn_output, self.codes)
        output_states += residual

        output_states = (comp_score * output_states).sum(dim=1)

        return output_states

    def forward_preln_ni(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = decoder_query
        comp_score = self.t(residual)

        decoder_query = self.layer_norm_1(decoder_query)
        decoder_keys = self.layer_norm_1(decoder_keys)
        self_attn_output = self.first_sub_layer(decoder_query.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_mask, self.codes, comp_score)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        enc_dec_attn_output = self.second_sub_layer(self_attn_output.unsqueeze(1), encoder_states.unsqueeze(1), encoder_states.unsqueeze(1), encoder_mask, self.codes, comp_score)
        enc_dec_attn_output += residual

        residual = enc_dec_attn_output
        enc_dec_attn_output = self.layer_norm_3(enc_dec_attn_output)
        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += residual

        return output_states

    def forward_postln_ni_full(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        comp_score = self.t(decoder_query)

        self_attn_output = self.first_sub_layer(decoder_query.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_mask, self.codes, None)
        self_attn_output += decoder_query.unsqueeze(1)

        self_attn_output = self.layer_norm_1(self_attn_output)

        enc_dec_attn_output = self.second_sub_layer(self_attn_output, encoder_states.unsqueeze(1), encoder_states.unsqueeze(1), encoder_mask, self.codes, None)
        enc_dec_attn_output += self_attn_output
        enc_dec_attn_output = self.layer_norm_2(enc_dec_attn_output)

        output_states = self.third_sub_layer(enc_dec_attn_output, self.codes)
        output_states += enc_dec_attn_output
        output_states = self.layer_norm_3(output_states)

        output_states = (comp_score * output_states).sum(dim=1)

        return output_states

    def forward_postln_ni(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        comp_score = self.t(decoder_query)

        self_attn_output = self.first_sub_layer(decoder_query.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_keys.unsqueeze(1), decoder_mask, self.codes, comp_score)
        self_attn_output += decoder_query

        self_attn_output = self.layer_norm_1(self_attn_output)

        enc_dec_attn_output = self.second_sub_layer(self_attn_output.unsqueeze(1), encoder_states.unsqueeze(1), encoder_states.unsqueeze(1), encoder_mask, self.codes, comp_score)
        enc_dec_attn_output += self_attn_output
        enc_dec_attn_output = self.layer_norm_2(enc_dec_attn_output)

        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += enc_dec_attn_output
        output_states = self.layer_norm_3(output_states)

        return output_states

    def forward_preln(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask, comp_score=None):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = decoder_query

        decoder_query = self.layer_norm_1(decoder_query)
        decoder_keys = self.layer_norm_1(decoder_keys)
        self_attn_output = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        enc_dec_attn_output = self.second_sub_layer(self_attn_output, encoder_states, encoder_states, encoder_mask)
        enc_dec_attn_output += residual

        residual = enc_dec_attn_output
        enc_dec_attn_output = self.layer_norm_3(enc_dec_attn_output)
        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += residual

        return output_states

    def forward_postln(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask, comp_score=None):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """

        self_attn_output = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
        self_attn_output += decoder_query

        self_attn_output = self.layer_norm_1(self_attn_output)

        enc_dec_attn_output = self.second_sub_layer(self_attn_output, encoder_states, encoder_states, encoder_mask)
        enc_dec_attn_output += self_attn_output
        enc_dec_attn_output = self.layer_norm_2(enc_dec_attn_output)

        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += enc_dec_attn_output
        output_states = self.layer_norm_3(output_states)

        return output_states

    def forward(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        return self.forward_call(decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask)

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
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
        pre_ln_final_layer_norm: bool = True,
        embedding: bool = False
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        layer = TransformerDecoderBlock(
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
        self.diagonal = 0

    def _get_memory_states(self, decoder_states, decoder_mems_list=None, i=0):
        if decoder_mems_list is not None:
            inp1 = torch.transpose(decoder_mems_list[i], 1, 2)  # Putting seq_len to last dim to handle export cases
            inp2 = torch.transpose(decoder_states, 1, 2)
            memory_states = torch.cat((inp1, inp2), dim=2)
            memory_states = torch.transpose(memory_states, 1, 2)  # Transposing back
        else:
            memory_states = decoder_states
        return memory_states

    def forward(
        self,
        decoder_states,
        decoder_mask,
        encoder_states,
        encoder_mask,
        decoder_mems_list=None,
        return_mems=False,
        return_mems_as_list=True,
    ):
        """
        Args:
            decoder_states: output of the embedding layer (B x L_dec x H)
            decoder_mask: decoder inputs mask (B x L_dec)
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            decoder_mems_list: list of the cached decoder hidden states
                for fast autoregressive generation which will be used instead
                of decoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all decoder layers
                or the last layer only
            return_mems_as_list: bool, when True, mems returned are as a list; otherwise mems are Tensor
        """
        decoder_attn_mask = form_attention_mask(decoder_mask, diagonal=self.diagonal)
        encoder_attn_mask = form_attention_mask(encoder_mask)
        memory_states = self._get_memory_states(decoder_states, decoder_mems_list, 0)
        if return_mems_as_list:
            cached_mems_list = [memory_states]
        else:
            cached_mems_list = memory_states.unsqueeze(0)

        for i, layer in enumerate(self.layers):
            decoder_states = layer(decoder_states, decoder_attn_mask, memory_states, encoder_states, encoder_attn_mask)
            memory_states = self._get_memory_states(decoder_states, decoder_mems_list, i + 1)
            if return_mems_as_list:
                cached_mems_list.append(memory_states)
            else:
                cached_mems_list = torch.cat((cached_mems_list, memory_states.unsqueeze(0)), dim=0)

        if self.final_layer_norm is not None:
            decoder_states = self.final_layer_norm(decoder_states)
            memory_states = self._get_memory_states(decoder_states, decoder_mems_list, i + 2)
            if return_mems_as_list:
                cached_mems_list.append(memory_states)
            else:
                cached_mems_list = torch.cat((cached_mems_list, memory_states.unsqueeze(0)), dim=0)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(max_batch, max_dim, 1024), device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=(max_batch, max_dim), device=sample.device)
        return tuple([input_ids, encoder_mask, input_ids, encoder_mask])
