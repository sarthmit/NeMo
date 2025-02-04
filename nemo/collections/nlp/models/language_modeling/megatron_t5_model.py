# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import math

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.utils import logging

__all__ = ["MegatronT5Model"]


class MegatronT5Model(MegatronLMEncoderDecoderModel):
    """
    Megatron T5 pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # validate cfg
        self._validate_cfg()

    @property
    def model_name(self):
        """Allows child classes to implement models with different data regime"""
        return "T5"

    def _validate_cfg(self):
        """Class-specific cfg validation"""
        # Make sure the user specifies dataset type as either 't5' or 't5_prefix_lm' only.
        if self._cfg.data.get('dataset_type', None) is not None:
            if self._cfg.data.get('dataset_type') not in ['t5', 't5_prefix_lm', 'ul2']:
                raise ValueError(
                    f"dataset_type must be either 't5', 't5_prefix_lm' or 'ul2'. found {self._cfg.data.get('dataset_type')}"
                )

        if hasattr(self._cfg.data, 'seq_length_dec') and self._cfg.data.get('dataset_type') == 't5':
            if self._cfg.data.seq_length_dec < self._cfg.data.seq_length * self._cfg.data.masked_lm_prob:
                raise ValueError(
                    f"Cannot have decoder max sequence length ({self._cfg.data.seq_length_dec}) less than encoder sequence length ({self._cfg.data.seq_length}) * masked_lm_prob ({self._cfg.data.masked_lm_prob})"
                )

        if self._cfg.data.get("dataset_type", "t5") == "ul2":
            if self._cfg.data.seq_length_dec != self._cfg.data.seq_length:
                raise ValueError(
                    f"Encoder and decoder sequence lengths must be the same while using the UL2 dataset type. Found encoder length {self._cfg.data.seq_length} and decoder length {self._cfg.data.seq_length_dec}"
                )
            if (
                self._cfg.tokenizer.num_sentinel_tokens
                < self._cfg.data.seq_length * self._cfg.data.extreme_masked_lm_prob
            ):
                raise ValueError(
                    f"Not enough sentinel tokens specified. Need at least {math.ceil(self._cfg.data.seq_length * self._cfg.data.extreme_masked_lm_prob)} sentinel tokens. Found {self._cfg.tokenizer.num_sentinel_tokens}"
                )

    @property
    def _build_train_valid_test_datasets_kwargs(self):
        """allows child classes to add kwargs to dataset building"""
        return dict(max_seq_length_dec=self._cfg.data.seq_length_dec,)

    def _build_vocab(self):
        self._add_special_tokens_to_tokenizer()

        super()._build_vocab()

    def _add_special_tokens_to_tokenizer(self):
        # T5-related construction
        self.num_sentinel_tokens = self._cfg.tokenizer.num_sentinel_tokens

        if self._cfg.tokenizer.library == 'huggingface' or self._cfg.tokenizer.library == 'megatron':
            additional_tokens = {
                'additional_special_tokens': [f'<extra_id_{i}>' for i in range(self.num_sentinel_tokens)]
            }
            if self._cfg.data.get("dataset_type", "t5") == "ul2":
                for mask_type in ['r', 's', 'x']:
                    additional_tokens['additional_special_tokens'].extend([f'<extra_id_{mask_type}>'])
            self.tokenizer.add_special_tokens(additional_tokens)

        if self._cfg.tokenizer.library == 'sentencepiece':
            # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
            # If cls, sep and mask are not attributes of the tokenizer, add it.
            if not hasattr(self.tokenizer, 'cls_token'):
                self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            if not hasattr(self.tokenizer.tokenizer, 'sep_id'):
                self.tokenizer.add_special_tokens({'sep_token': '<sep>'})
            if not hasattr(self.tokenizer.tokenizer, 'mask_id'):
                self.tokenizer.add_special_tokens({'mask_token': '<mask>'})

            # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
            if not hasattr(self.tokenizer, 'pad_token'):
                if hasattr(self.tokenizer.tokenizer, 'pad_id') and self.tokenizer.tokenizer.pad_id() > 0:
                    self.tokenizer.pad_token = self.tokenizer.tokenizer.id_to_piece(self.tokenizer.tokenizer.pad_id())
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

            if not hasattr(self.tokenizer, 'bos_token'):
                if hasattr(self.tokenizer.tokenizer, 'bos_id') and self.tokenizer.tokenizer.bos_id() > 0:
                    self.tokenizer.bos_token = self.tokenizer.tokenizer.id_to_piece(self.tokenizer.tokenizer.bos_id())
                else:
                    self.tokenizer.add_special_tokens({'bos_token': '<bos>'})
            else:
                self.tokenizer.add_special_tokens({'bos_token': '<s>'})

            if not hasattr(self.tokenizer, 'eos_token'):
                if hasattr(self.tokenizer.tokenizer, 'eos_id') and self.tokenizer.tokenizer.eos_id() > 0:
                    self.tokenizer.eos_token = self.tokenizer.tokenizer.id_to_piece(self.tokenizer.tokenizer.eos_id())
                else:
                    self.tokenizer.add_special_tokens({'eos_token': '<eos>'})
            else:
                self.tokenizer.add_special_tokens({'eos_token': '</s>'})

            # Special check to see if <extra_id_{}> is already present in the tokenizer. If it is, only modify the additional_special_tokens function.
            for i in range(self.num_sentinel_tokens):
                if f'▁<extra_id_{i}>' in self.tokenizer.vocab:
                    self.tokenizer.special_token_to_id[f'<extra_id_{i}>'] = self.tokenizer.text_to_ids(
                        f'<extra_id_{i}>'
                    )[0]
                else:
                    self.tokenizer.add_special_tokens([f'<extra_id_{i}>'])

            if self._cfg.data.get("dataset_type", "t5") == "ul2":
                for mask_type in ['r', 's', 'x']:
                    if f'▁<extra_id_{mask_type}>' in self.tokenizer.vocab:
                        self.tokenizer.special_token_to_id[f'<extra_id_{i}>'] = self.tokenizer.text_to_ids(
                            f'<extra_id_{i}>'
                        )[0]
                    else:
                        self.tokenizer.add_special_tokens([f'<extra_id_{mask_type}>'])

    def build_train_valid_test_datasets(self):
        logging.info(f'Building {self.model_name} datasets.')
        global_batch_size = self._cfg.global_batch_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self._cfg,
            trainer=self.trainer,
            tokenizer=self.tokenizer,
            data_prefix=self._cfg.data.data_prefix,
            data_impl=self._cfg.data.data_impl,
            splits_string=self._cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self._cfg.data.seq_length,
            masked_lm_prob=self._cfg.data.masked_lm_prob,
            short_seq_prob=self._cfg.data.short_seq_prob,
            seed=self._cfg.seed,
            skip_warmup=self._cfg.data.skip_warmup,
            dataset_type=self._cfg.data.get('dataset_type', self.model_name.lower()),
            max_ngram_size=self._cfg.data.get('max_ngram_size', 10),
            mean_ngram_size=self._cfg.data.get('mean_ngram_size', None),
            geometric_dist=self._cfg.data.get('geometric_dist', True),
            permutation=self._cfg.data.get('permutation', False),
            whole_word_masking=self._cfg.data.get('whole_word_masking', True),
            favor_long_ngrams=self._cfg.data.get('favor_long_ngrams', False),
            respect_document_boundaries=self._cfg.data.get('respect_document_boundaries', True),
            # additional arguments from child classes
            **self._build_train_valid_test_datasets_kwargs,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building {self.model_name} datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def list_available_models(self):
        pass
