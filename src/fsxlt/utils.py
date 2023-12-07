import json
import logging
import math
import random

from typing import Dict, Optional

import numpy as np
import torch

import datasets
from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader,
    Sampler,
)

from transformers import (
    AdapterConfig
)

from sft import (
    load_multisource_dataset,
    MultiSourceDataLoader,
    MultiSourceDataset,
    SFT,
)
from sft.multisource import BATCH_SOURCE_KEY
import transformers.adapters.composition as ac

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def load_few_shot_data(
    fs_args,
    training_args,
    **kwargs,
):
    dataset_descriptor = {}
    adapter_map = {}
    if fs_args.multisource_data is not None:
        with open(fs_args.multisource_data) as f:
            dataset_descriptor = json.load(f)
            if fs_args.source_lang is not None:
                if fs_args.source_num_examples == 0:
                    dataset_descriptor.pop(fs_args.source_lang, None)
                elif fs_args.source_num_examples > 0:
                    dataset_descriptor[fs_args.source_lang]['data']['max_train_samples'] = fs_args.source_num_examples
                if not fs_args.include_source_in_evaluation:
                    dataset_descriptor[fs_args.source_lang]['data']['validation_split'] = None
                    dataset_descriptor[fs_args.source_lang]['data']['test_split'] = None
            for lang, lang_descriptor in dataset_descriptor.items():
                if fs_args.source_lang is not None and lang != fs_args.source_lang and not lang.endswith('translated'):
                    if fs_args.target_num_examples == 0:
                        dataset_descriptor.pop(lang, None)
                    elif fs_args.target_num_examples > 0 and 'max_train_samples' not in dataset_descriptor[lang]['data']:
                        dataset_descriptor[lang]['data']['max_train_samples'] = fs_args.target_num_examples
                    if fs_args.target_train_split and 'train_split' not in dataset_descriptor[lang]['data']:
                        dataset_descriptor[lang]['data']['train_split'] = fs_args.target_train_split
                    if not fs_args.include_target_in_evaluation:
                        dataset_descriptor[lang]['data']['validation_split'] = None
                        dataset_descriptor[lang]['data']['test_split'] = None
                    if 'train_upsampling' not in dataset_descriptor[lang]:
                        dataset_descriptor[lang]['train_upsampling'] = fs_args.target_upsampling
                if fs_args.language_adaptation:
                    adapter_name = lang.split('_')[0]
                    adapter_map[lang] = adapter_name
    else:
        if fs_args.source_lang is not None and fs_args.source_num_examples != 0:
            if fs_args.source_multisource_data is not None:
                with open(fs_args.source_multisource_data) as f:
                    data_descriptor = json.load(f)
                for data_name, descriptor in data_descriptor.items():
                    full_data_name = f'{fs_args.source_lang}_{data_name}'
                    dataset_descriptor[full_data_name] = descriptor
                    if fs_args.source_adaptation:
                        adapter_map[full_data_name] = fs_args.source_lang
            else:
                data_descriptor = {}
                if fs_args.source_dataset_name is not None:
                    data_descriptor['name'] = fs_args.source_dataset_name
                if fs_args.source_config_name is not None:
                    data_descriptor['config_name'] = fs_args.source_config_name
                if fs_args.source_train_file is not None:
                    data_descriptor['train_file'] = fs_args.source_train_file
                if fs_args.source_validation_file is not None:
                    data_descriptor['validation_file'] = fs_args.source_validation_file
                if fs_args.source_test_file is not None:
                    data_descriptor['test_file'] = fs_args.source_test_file
                if fs_args.source_num_examples > 0:
                    data_descriptor['max_train_samples'] = fs_args.source_num_examples
                if not fs_args.include_source_in_evaluation:
                    data_descriptor['validation_split'] = None
                    data_descriptor['test_split'] = None
                source_descriptor = {'data': data_descriptor}
                dataset_descriptor[fs_args.source_lang] = source_descriptor
                if fs_args.source_adaptation:
                    adapter_map[fs_args.source_lang] = fs_args.source_lang
        if fs_args.target_lang is not None and fs_args.target_num_examples != 0:
            if fs_args.target_multisource_data is not None:
                with open(fs_args.target_multisource_data) as f:
                    data_descriptor = json.load(f)
                for data_name, descriptor in data_descriptor.items():
                    full_data_name = f'{fs_args.target_lang}_{data_name}'
                    dataset_descriptor[full_data_name] = descriptor
                    if fs_args.target_adaptation:
                        adapter_map[full_data_name] = fs_args.target_lang
            else:
                data_descriptor = {}
                if fs_args.target_dataset_name is not None:
                    data_descriptor['name'] = fs_args.target_dataset_name
                if fs_args.target_config_name is not None:
                    data_descriptor['config_name'] = fs_args.target_config_name
                if fs_args.target_train_file is not None:
                    data_descriptor['train_file'] = fs_args.target_train_file
                if fs_args.target_validation_file is not None:
                    data_descriptor['validation_file'] = fs_args.target_validation_file
                if fs_args.target_test_file is not None:
                    data_descriptor['test_file'] = fs_args.target_test_file
                if fs_args.target_train_split:
                    data_descriptor['train_split'] = fs_args.target_train_split
                if not fs_args.include_target_in_evaluation:
                    data_descriptor['validation_split'] = None
                    data_descriptor['test_split'] = None
                if fs_args.target_num_examples > 0:
                    data_descriptor['max_train_samples'] = fs_args.target_num_examples
                target_descriptor = {
                    'data': data_descriptor,
                    'train_upsampling': fs_args.target_upsampling
                }
                dataset_descriptor[fs_args.target_lang] = target_descriptor
                if fs_args.target_adaptation:
                    adapter_map[fs_args.target_lang] = fs_args.target_lang

    raw_datasets, _ = load_multisource_dataset(
        dataset_descriptor,
        training_args,
        **kwargs,
    )
    return raw_datasets, adapter_map


def load_sfts(fs_args, adapter_map):
    lang_to_sft = {}
    if not fs_args.lang_adapt_format and adapter_map:
        raise RuntimeError(
            'When language adaptation is required, --lang_adapt_format must be specified.'
        )

    for lang in set(adapter_map.values()):
        lang_to_sft[lang] = SFT(
            fs_args.lang_adapt_format.format(lang)
        )
    return {
        k: lang_to_sft[v] for k, v in adapter_map.items()
    }

def set_adapter_args(fs_args, adapter_args):
    adapter_args.train_adapter = True
    adapter_args.task_adapter_drop_last = "yes"
    if not adapter_args.load_adapter:
        adapter_args.adapter_config = "pfeiffer"
        adapter_args.adapter_reduction_factor = 16
    if fs_args.source_adaptation or fs_args.target_adaptation or fs_args.language_adaptation:
        adapter_args.load_lang_adapter = True
        adapter_args.lang_adapter_drop_last = "yes"

def load_adapters(fs_args, model, adapter_args, task_name, adapter_map):
    assert fs_args.adaptation_method == "adapters"
    if not fs_args.lang_adapt_format and adapter_map:
        raise RuntimeError(
            'When language adaptation is required, --lang_adapt_format must be specified.'
        )
    set_adapter_args(fs_args, adapter_args)
    lang_adapters = {}
    # Setup adapters
    if adapter_args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
                leave_out=[11] if adapter_args.task_adapter_drop_last == "yes" else [],
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                    leave_out=[11] if adapter_args.task_adapter_drop_last == "yes" else [],
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        # optionally load a pre-trained language adapter]

        lang_adapter_name = None
        if adapter_args.load_lang_adapter:
            for source, lang in adapter_map.items():
                # resolve the language adapter config
                lang_adapter_config = AdapterConfig.load(
                    adapter_args.lang_adapter_config,
                    non_linearity=adapter_args.lang_adapter_non_linearity,
                    reduction_factor=adapter_args.lang_adapter_reduction_factor,
                )
                # load the language adapter from Hub
                lang_adapter_name = model.load_adapter(
                    fs_args.lang_adapt_format.format(lang),
                    config=lang_adapter_config,
                    load_as=source,
                    leave_out=[11] if adapter_args.lang_adapter_drop_last else [],
                )
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        else:
            model.set_active_adapters([task_name])

    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )
    return adapter_map.keys()

def MultiSourcePlugin(_Trainer):

    class _MultiSourceTrainer(_Trainer):
        
        def __init__(
            self,
            *args,
            train_dataset=None,
            eval_dataset=None,
            source_sfts=None,
            source_adapters=None,
            source_sft_apply_abs=False,
            **kwargs
        ):
            self._multisource = (
                (train_dataset is not None and isinstance(train_dataset, MultiSourceDataset)) or
                (eval_dataset is not None and isinstance(eval_dataset, MultiSourceDataset))
            )

            if source_sfts is not None and source_adapters is not None:
                raise ValueError('You cannot use both SFTs and Adapters at the some time.')

            if self._multisource and source_sfts is None and source_adapters is None:
                raise ValueError('Multi-source datasets provided, but no source SFTs')
            self._source_sfts = source_sfts
            self._source_adapters = source_adapters
            self._source_sft_apply_abs = source_sft_apply_abs
            self._activated_sft = None

            super().__init__(
                *args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                **kwargs,
            )
            self.training = False

        def activate_sft(self, source):
            if source == self._activated_sft:
                return

            if self._activated_sft is not None:
                activated_sft = self._source_sfts[self._activated_sft]
                activated_sft.revert(self.model)

            if source is not None:
                sft_to_activate = self._source_sfts[source]
                sft_to_activate.apply(self.model, with_abs=self._source_sft_apply_abs)
            
            self._activated_sft = source

        def activate_adapter(self, source):
            if len(self.model.active_adapters) != 2:
                raise ValueError(f"Need two adapters (language and task): found {len(self.model.active_adapters)}")
            if self.training:
                self.model.train_adapter([self.model.active_adapters[1]])
                self.model.set_active_adapters(ac.Stack(source, self.model.active_adapters[0]))
            else:
                self.model.set_active_adapters(ac.Stack(source, self.model.active_adapters[1]))

        def adapt(self, inputs):
            if self._source_sfts is None and self._source_adapters is None:
                return

            source = inputs.pop(BATCH_SOURCE_KEY, None)
            if source is None:
                raise ValueError(f'Batch contained no key "{BATCH_SOURCE_KEY}"')

            if self._source_sfts is not None:
                if source in self._source_sfts:
                    self.activate_sft(source)

            if self._source_adapters is not None:
                if source in self._source_adapters:
                    self.activate_adapter(source)

        def deadapt(self):
            if self._source_sfts is not None:
                self.activate_sft(None)

            if self._source_adapters is not None:
                pass

        def training_step(self, model, inputs):
            self.training = True
            self.adapt(inputs)
            output = super().training_step(model, inputs)
            self.deadapt()
            return output

        def prediction_step(self, model, inputs, *args, **kwargs):
            self.training = False
            self.adapt(inputs)
            output = super().prediction_step(model, inputs, *args, **kwargs)
            self.deadapt()
            return output

        def get_train_dataloader(self) -> DataLoader:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            if isinstance(train_dataset, MultiSourceDataset):
                loaders = {}
                for source, dataset in train_dataset.datasets.items():
                    self.train_dataset = dataset
                    loaders[source] = super().get_train_dataloader()
                self.train_dataset = train_dataset
                return MultiSourceDataLoader(
                    train_dataset,
                    loaders,
                    sampling_policy='random',
                )
            else:
                return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

            if isinstance(eval_dataset, MultiSourceDataset):
                loaders = {}
                for source, dataset in eval_dataset.datasets.items():
                    loaders[source] = super().get_eval_dataloader(eval_dataset=dataset)
                return MultiSourceDataLoader(
                    eval_dataset,
                    loaders,
                    sampling_policy='sequential',
                )
            else:
                return super().get_eval_dataloader(eval_dataset)

        def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
            if isinstance(test_dataset, MultiSourceDataset):
                loaders = {}
                for source, dataset in test_dataset.datasets.items():
                    loaders[source] = super().get_test_dataloader(dataset)
                return MultiSourceDataLoader(
                    test_dataset,
                    loaders,
                    sampling_policy='sequential',
                )
            else:
                return super().get_test_dataloader(test_dataset)

    return _MultiSourceTrainer
