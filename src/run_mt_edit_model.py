#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from torch import nn

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import ClassLabel, load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from modeling_bert import BertForMTSparseEditModel
from modeling_bert import BertForMTSparseLastLayerEditModel
from modeling_bert import BertForMTSparseFFDEditModel

from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from data_collator import DataCollatorForLanguageModeling
from accelerate import DistributedDataParallelKwargs
import copy

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.29.0.dev0")

logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--tagging_file_list", nargs='+', default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--generation_file_list", nargs='+', default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_tagging_file_list", nargs='+', default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_generation_file_list", nargs='+', default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--prediction_tagging_file", type=str, default=None, help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--prediction_generation_file", type=str, default=None, help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--mask_column_name",
        type=str,
        default=None,
        help="The column name of mask to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help=".",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help=".",
    )
    parser.add_argument(
        "--only_do_predict",
        action="store_true",
        help=".",
    )
    parser.add_argument(
        "--token_num_per_slot",
        type=int,
        default=4,
        help="Number of tokens for one slot.",
    )
    parser.add_argument(
        "--sum_tasks",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "--sparse_mode",
        type=str,
        default='cls_head',
        help="'cls_head', 'last_layer', 'ffd'",
    )
    parser.add_argument(
        "--sparse_encdec",
        type=bool,
        default=False,
        help="use sparse layer or not",
    )
    parser.add_argument(
        "--gate_type",
        type=str,
        default='task_id',
        help="task_id moe mmoe",
    )
    parser.add_argument(
        "--head_num",
        type=int,
        default=4,
        help=" ",
    )
    parser.add_argument(
        "--sparse_level",
        type=str,
        default='sentence_level',
        help="sentence_level token_level",
    )
    parser.add_argument(
        "--gate_temperature",
        type=float,
        default=1,
        help=" ",
    )
    parser.add_argument(
        "--norm_type",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="",
    )
    parser.add_argument(
        "--min_pro",
        type=float,
        default=0,
        help="",
    )
    parser.add_argument(
        "--is_additional_finetune",
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "--is_freeze",
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "--copy_ids", 
        nargs='+', 
        default=None, 
        help=""
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )
    args = parser.parse_args()

    # Sanity checks
    # if args.task_name is None and args.train_file is None and args.validation_file is None and args.tagging_file is None and args.generation_file is None:
    #     raise ValueError("Need either a task name or a training/validation file.")
    # else:
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if args.tagging_file_list is not None:
        extension = args.tagging_file_list[0].split(".")[-1]
        assert extension in ["csv", "json"], "`tagging_file` should be a csv or a json file."
    if args.generation_file_list is not None:
        extension = args.generation_file_list[0].split(".")[-1]
        assert extension in ["csv", "json"], "`generation_file` should be a csv or a json file."
    if args.prediction_tagging_file is not None:
        extension = args.prediction_tagging_file.split(".")[-1]
        assert extension in ["csv", "json"], "`prediction_tagging_file` should be a csv or a json file."
    if args.prediction_generation_file is not None:
        extension = args.prediction_generation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`prediction_generation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_ner_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir, kwargs_handlers=[ddp_kwargs]) if args.with_tracking else Accelerator(kwargs_handlers=[ddp_kwargs])
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        tagging_raw_datasets_list = []
        if args.tagging_file_list is not None and args.validation_tagging_file_list is not None:
            for tagging_file, validation_tagging_file in zip(args.tagging_file_list, args.validation_tagging_file_list):
                tagging_data_files = {}
                tagging_data_files["train"] = tagging_file
                tagging_data_files["validation"] = validation_tagging_file
                extension = tagging_file.split(".")[-1]
                tagging_raw_datasets = load_dataset(extension, data_files=tagging_data_files)
                tagging_raw_datasets_list.append(tagging_raw_datasets)

        generation_raw_datasets_list = []
        if args.generation_file_list is not None and args.validation_generation_file_list is not None:
            for generation_file, validation_generation_file in zip(args.generation_file_list, args.validation_generation_file_list):
                generation_data_files = {}
                generation_data_files["train"] = generation_file
                generation_data_files["validation"] = validation_generation_file
                extension = generation_file.split(".")[-1]
                generation_raw_datasets = load_dataset(extension, data_files=generation_data_files)
                generation_raw_datasets_list.append(generation_raw_datasets)

    # Trim a number of training examples
    if args.debug:
        for split in tagging_raw_datasets.keys():
            tagging_raw_datasets[split] = tagging_raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if args.tagging_file_list is not None:
        column_names = tagging_raw_datasets_list[0]["train"].column_names
        features = tagging_raw_datasets_list[0]["train"].features
    else:
        column_names = generation_raw_datasets_list[0]["train"].column_names
        # features = generation_raw_datasets["train"].features
        features = None

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    if args.mask_column_name is not None:
        mask_column_name = args.mask_column_name
    else:
        mask_column_name = 'mask'
    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    
    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    num_labels = 4 # set an arbitrary num_labels for init. (when there is no tagging task.)
    if args.tagging_file_list is not None and not args.is_additional_finetune:
        labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
        if labels_are_int:
            label_list = features[label_column_name].feature.names
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            labels_ = []
            for tagging_raw_datasets in tagging_raw_datasets_list:
                labels_.extend(tagging_raw_datasets["train"][label_column_name])
            label_list = get_label_list(labels_)
            label_to_id = {l: i for i, l in enumerate(label_list)}

        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.is_additional_finetune:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        label_to_id = config.label2id
        id_to_label = config.id2label
        num_labels = len(label_to_id)
        config.num_labels = num_labels
        label_list = list(label_to_id.keys())
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if not args.only_do_predict:
        config.sum_tasks = args.sum_tasks
    if not args.only_do_predict and args.gate_type == 'task_id':
        assert args.head_num == args.sum_tasks, 'heads are routed by task_id, so the sum of them should be the same.'
    config.head_num = args.head_num
    if args.sparse_encdec:
        config.sum_layers = config.head_num*2
    else:
        config.sum_layers = config.head_num
    config.sparse_mode = args.sparse_mode
    config.gate_type = args.gate_type
    config.sparse_level = args.sparse_level
    config.sparse_encdec = args.sparse_encdec
    config.gate_temperature = args.gate_temperature

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if args.sparse_mode == 'cls_head':
        ModelClass = BertForMTSparseEditModel
    elif args.sparse_mode == 'last_layer':
        ModelClass = BertForMTSparseLastLayerEditModel
        args.ignore_mismatched_sizes = True
    elif args.sparse_mode == 'ffd':
        ModelClass = BertForMTSparseFFDEditModel
        args.ignore_mismatched_sizes = True

    if args.model_name_or_path:
        model = ModelClass.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes or args.is_additional_finetune,
        )
    else:
        logger.info("Training new model from scratch")
        model = ModelClass.from_config(config)

    # def clone_linear(init_module, module):
    #     module.weight = init_module.weight.clone()
    #     module.bias = init_module.bias.clone()

    # def clone_bert_attention(init_module, module):
    #     clone_linear(module.attention.self.query, init_module.attention.self.query)
    #     clone_linear(module.attention.self.key, init_module.attention.self.key)
    #     clone_linear(module.attention.self.value, init_module.attention.self.value)
    #     module.attention.self.dropout = init_module.attention.self.dropout
    #     module.attention.self.distance_embedding.weight = init_module.attention.self.distance_embedding.weight.clone()

    #     clone_linear(module.attention.output.dense, init_module.attention.output.dense)
    #     module.attention.output.LayerNorm = init_module.attention.output.LayerNorm
    #     module.attention.output.dropout = init_module.attention.output.dropout

    # def clone_bert_layer(init_module, module):
    #     clone_bert_attention(module.attention, init_module.attention)
    def clone_module(polyak_factor, target_network, network):
        # update target net
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))
    
    if args.sparse_mode == 'last_layer' and args.do_train and not args.is_additional_finetune:
        # print(len(model.bert.encoder.layer))
        init_module = model.bert.encoder.layer[-1]
        # print(init_module.intermediate.dense.weight[0][:10])
        if args.gate_type == 'task_id':
            for module in model.bert.encoder.sparse_layer:
                # print('before: ',module.intermediate.dense.weight[0][:10])
                clone_module(1, module, init_module)
                # print('after: ', module.intermediate.dense.weight[0][:10])  
        elif args.gate_type == 'moe'or args.gate_type == 'mmoe':
            for module in model.bert.encoder.enc_sparse_layer:
                clone_module(1, module, init_module)
            for module in model.bert.encoder.dec_sparse_layer:
                clone_module(1, module, init_module)
    if args.sparse_mode == 'ffd' and args.do_train and not args.is_additional_finetune:
        for layer in model.bert.encoder.layer:
            init_module_inter = layer.intermediate
            init_module_out = layer.output
            for module_inter, module_out in zip(layer.enc_intermediate_layer, layer.enc_output_layer):
                clone_module(1, module_inter, init_module_inter)
                clone_module(1, module_out, init_module_out)
            for module_inter, module_out in zip(layer.dec_intermediate_layer, layer.dec_output_layer):
                clone_module(1, module_inter, init_module_inter)
                clone_module(1, module_out, init_module_out)

    if args.sparse_mode != 'cls_head' and args.gate_type == 'moe' and args.do_train and not args.is_additional_finetune:
        if args.sparse_mode == 'last_layer':
            enc_gate = model.bert.encoder.enc_gate
            dec_gate = model.bert.encoder.dec_gate
            torch.nn.init.normal_(enc_gate.weight.data, mean=0., std=0.001)
            torch.nn.init.normal_(dec_gate.weight.data, mean=0., std=0.001)
        elif args.sparse_mode == 'ffd':
            for layer in model.bert.encoder.layer:
                enc_gate = layer.enc_gate
                dec_gate = layer.dec_gate
                torch.nn.init.normal_(enc_gate.weight.data, mean=0., std=0.001)
                torch.nn.init.normal_(dec_gate.weight.data, mean=0., std=0.001)
    elif args.sparse_mode != 'cls_head' and args.gate_type == 'mmoe' and args.do_train and not args.is_additional_finetune:
        if args.sparse_mode == 'last_layer':
            enc_gates = model.bert.encoder.enc_gate_list
            dec_gates = model.bert.encoder.dec_gate_list
            for enc_gate in enc_gates:
                torch.nn.init.normal_(enc_gate.weight.data, mean=0., std=0.001)
            for dec_gate in dec_gates:
                torch.nn.init.normal_(dec_gate.weight.data, mean=0., std=0.001)
        elif args.sparse_mode == 'ffd':
            for layer in model.bert.encoder.layer:
                enc_gates = layer.enc_gate_list
                dec_gates = layer.dec_gate_list
                for enc_gate in enc_gates:
                    torch.nn.init.normal_(enc_gate.weight.data, mean=0., std=0.001)
                for dec_gate in dec_gates:
                    torch.nn.init.normal_(dec_gate.weight.data, mean=0., std=0.001)

    if args.do_train and not args.is_additional_finetune:
        special_tokens_dict = {'additional_special_tokens': ['[DELETE]', '[/DELETE]', '$START', '[TRANSFORM_VERB]', '[/TRANSFORM_VERB]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    config.vocab_size = len(tokenizer)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if args.generation_file_list is not None and args.do_train and not args.is_additional_finetune:
        for mlm_cls in model.mlm_cls_list:
            mlm_cls.predictions.decoder.weight = model.get_input_embeddings().weight # decoder is not sparse
            mlm_cls.predictions.decoder.bias.data = nn.functional.pad(
                mlm_cls.predictions.decoder.bias.data,
                (
                    0,
                    mlm_cls.predictions.decoder.weight.shape[0] - mlm_cls.predictions.decoder.bias.shape[0],
                ),
                "constant",
                0,
            )
            # mlm_cls.predictions.decoder.weight = nn.Parameter(model.get_input_embeddings().weight.clone())
        
    if args.is_additional_finetune:
        if args.sparse_mode == 'ffd' and args.do_train:
            for layer in model.bert.encoder.layer:
                copy_enc_intermediate_layer = []
                copy_enc_output_layer = []
                copy_dec_intermediate_layer = []
                copy_dec_output_layer = []
                for module_inter, module_out in zip(layer.enc_intermediate_layer, layer.enc_output_layer):
                    copy_enc_intermediate_layer.append(copy.deepcopy(module_inter))
                    copy_enc_output_layer.append(copy.deepcopy(module_out))
                for module_inter, module_out in zip(layer.dec_intermediate_layer, layer.dec_output_layer):
                    copy_dec_intermediate_layer.append(copy.deepcopy(module_inter))
                    copy_dec_output_layer.append(copy.deepcopy(module_out))

                for index, copy_index in enumerate(args.copy_ids):
                    copy_index = int(copy_index)
                    clone_module(1, layer.enc_intermediate_layer[index], copy_enc_intermediate_layer[copy_index])
                    clone_module(1, layer.enc_output_layer[index], copy_enc_output_layer[copy_index])
                    clone_module(1, layer.dec_intermediate_layer[index], copy_dec_intermediate_layer[copy_index])
                    clone_module(1, layer.dec_output_layer[index], copy_dec_output_layer[copy_index])
            if args.gate_type != 'token_id':
                pass
        else:
            pass

    if args.is_additional_finetune:
        copy_mlm_cls = []
        copy_cls = []
        for mlm_cls in model.mlm_cls_list:
            copy_mlm_cls.append(copy.deepcopy(mlm_cls))
        for cls_ in model.classifier_list:
            copy_cls.append(copy.deepcopy(cls_))

        for index, copy_index in enumerate(args.copy_ids):
            copy_index = int(copy_index)
            clone_module(1, model.mlm_cls_list[index], copy_mlm_cls[copy_index])
            clone_module(1, model.classifier_list[index], copy_cls[copy_index])


    # Model has labels -> use them.
    if args.tagging_file_list is not None and not args.is_additional_finetune:
        if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
            if sorted(model.config.label2id.keys()) == sorted(label_list):
                # Reorganize `label_list` to match the ordering of the model.
                if labels_are_int:
                    label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                    label_list = [model.config.id2label[i] for i in range(num_labels)]
                else:
                    label_list = [model.config.id2label[i] for i in range(num_labels)]
                    label_to_id = {l: i for i, l in enumerate(label_list)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                    f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
                )

        # Set the correspondences label/ID inside the model config
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = dict(enumerate(label_list))

    if args.tagging_file_list is not None:
        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_masks(examples):
        # print(examples[text_column_name])
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            # return_special_tokens_mask=True,
        )
        labels = []
        inputs = []
        token_type_ids_s = []
        attention_mask_s = []
        for i, label in enumerate(examples[mask_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            previous_label = None
            token_num = 0
            input_ids = []
            label_ids = []
            token_type_ids = []
            attention_mask = []
            for index, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                # print(tokenized_inputs.keys())

                if word_idx is None:
                    label_ids.append(-100)
                    # continue
                # We set the label for the first token of each word.
                elif previous_label != label[word_idx]:
                    if previous_label == 1 and token_num > 0 and token_num < args.token_num_per_slot:
                        label_ids.extend([tokenizer.pad_token_id for _ in range(args.token_num_per_slot - token_num)])
                        input_ids.extend([tokenizer.mask_token_id for _ in range(args.token_num_per_slot - token_num)])
                        token_type_ids.extend([0 for _ in range(args.token_num_per_slot - token_num)])
                        attention_mask.extend([1 for _ in range(args.token_num_per_slot - token_num)])
                        token_num = 0
                        label_ids.append(-100)

                    elif previous_label == 1 or (previous_label is None and label[word_idx] == 0):
                        token_num = 0
                        label_ids.append(-100)

                    else:
                        label_ids.append(tokenized_inputs["input_ids"][i][index])
                        token_num = 1
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if label[word_idx] == 1:
                        token_num += 1
                    if token_num > args.token_num_per_slot:
                        # token_num = 0
                        continue
                    if label[word_idx] == 1:
                        label_ids.append(tokenized_inputs["input_ids"][i][index])
                    else:
                        label_ids.append(-100)

                if word_idx is None:
                    input_ids.append(tokenized_inputs["input_ids"][i][index])
                elif label[word_idx] == 1:
                    input_ids.append(tokenizer.mask_token_id)
                else:
                    input_ids.append(tokenized_inputs["input_ids"][i][index])
                    
                token_type_ids.append(tokenized_inputs["token_type_ids"][i][index])
                attention_mask.append(tokenized_inputs["attention_mask"][i][index])
                if word_idx is not None:
                    previous_word_idx = word_idx
                    previous_label = label[word_idx]
            if args.pad_to_max_length:
                if len(label_ids) != len(input_ids) or len(input_ids) != len(token_type_ids) or len(token_type_ids) != len(attention_mask):
                    print(len(label_ids), len(input_ids), len(token_type_ids), len(attention_mask))
                    raise ValueError("The align algorithm is wrong.")
                if len(label_ids) < args.max_length:
                    label_ids.extend([-100 for _ in range(args.max_length - len(label_ids))])
                if len(input_ids) < args.max_length:
                    input_ids.extend([tokenizer.pad_token_id for _ in range(args.max_length - len(input_ids))])
                if len(token_type_ids) < args.max_length:
                    token_type_ids.extend([0 for _ in range(args.max_length - len(token_type_ids))])
                if len(attention_mask) < args.max_length:
                    attention_mask.extend([0 for _ in range(args.max_length - len(attention_mask))])
                labels.append(label_ids[:args.max_length])
                inputs.append(input_ids[:args.max_length])
                token_type_ids_s.append(token_type_ids[:args.max_length])
                attention_mask_s.append(attention_mask[:args.max_length])
            else:
                labels.append(label_ids)
                inputs.append(input_ids)
                token_type_ids_s.append(token_type_ids)
                attention_mask_s.append(attention_mask)
        tokenized_inputs["input_ids"] = inputs
        # tokenized_inputs["mask"] = labels
        tokenized_inputs["token_type_ids"] = token_type_ids_s
        tokenized_inputs["attention_mask"] = attention_mask_s
        tokenized_inputs["labels"] = labels
        # print(tokenized_inputs.keys())
        
        return tokenized_inputs

    with accelerator.main_process_first():
        if args.tagging_file_list is not None and args.do_train:
            train_tagging_dataset_list = []
            eval_tagging_dataset_list = []
            for tagging_raw_datasets in tagging_raw_datasets_list:
                processed_tagging_raw_datasets = tagging_raw_datasets.map(
                    tokenize_and_align_labels,
                    batched=True,
                    remove_columns=tagging_raw_datasets["train"].column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                train_tagging_dataset = processed_tagging_raw_datasets["train"]
                eval_tagging_dataset = processed_tagging_raw_datasets["validation"]
                train_tagging_dataset_list.append(train_tagging_dataset)
                eval_tagging_dataset_list.append(eval_tagging_dataset)

        if args.generation_file_list is not None and args.do_train:
            train_generation_dataset_list = []
            eval_generation_dataset_list = []
            for generation_raw_datasets in generation_raw_datasets_list:
                processed_generation_raw_datasets = generation_raw_datasets.map(
                    tokenize_and_align_masks,
                    batched=True,
                    remove_columns=generation_raw_datasets["train"].column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                train_generation_dataset = processed_generation_raw_datasets["train"]
                eval_generation_dataset = processed_generation_raw_datasets["validation"]
                train_generation_dataset_list.append(train_generation_dataset)
                eval_generation_dataset_list.append(eval_generation_dataset)


    # Log a few random samples from the training set:
    if args.tagging_file_list is not None and args.do_train:
        for index in random.sample(range(len(train_tagging_dataset_list[0])), 10):
            logger.info(f"Sample {index} of the training set: {train_tagging_dataset_list[0][index]}.")
    if args.generation_file_list is not None and args.do_train:
        for index in random.sample(range(len(train_generation_dataset_list[0])), 10):
            logger.info(f"Sample {index} of the training set: {train_generation_dataset_list[0][index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        tagging_data_collator = default_data_collator
        generation_data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        tagging_data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
        generation_data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_tagging_dataloader_list = []
    train_generation_dataloader_list = []
    eval_tagging_dataloader_list = []
    eval_generation_dataloader_list = []
    if args.tagging_file_list is not None and args.do_train:
        for train_tagging_dataset in train_tagging_dataset_list:
            train_tagging_dataloader = DataLoader(
                train_tagging_dataset, shuffle=True, collate_fn=tagging_data_collator, batch_size=args.per_device_train_batch_size
            )
            train_tagging_dataloader_list.append(train_tagging_dataloader)
        for eval_tagging_dataset in eval_tagging_dataset_list:
            eval_tagging_dataloader = DataLoader(eval_tagging_dataset, collate_fn=tagging_data_collator, batch_size=args.per_device_eval_batch_size)
            eval_tagging_dataloader_list.append(eval_tagging_dataloader)
    if args.generation_file_list is not None and args.do_train:
        for train_generation_dataset in train_generation_dataset_list:
            train_generation_dataloader = DataLoader(
                train_generation_dataset, shuffle=True, collate_fn=generation_data_collator, batch_size=args.per_device_train_batch_size
            )
            train_generation_dataloader_list.append(train_generation_dataloader)
        for eval_generation_dataset in eval_generation_dataset_list:
            eval_generation_dataloader = DataLoader(eval_generation_dataset, collate_fn=generation_data_collator, batch_size=args.per_device_eval_batch_size)
            eval_generation_dataloader_list.append(eval_generation_dataloader)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    if args.is_freeze:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in layer.enc_intermediate_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            } for layer in model.bert.encoder.layer
        ]
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.enc_output_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.dec_intermediate_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.dec_output_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in model.classifier_list.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            }
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.predictions.transform.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            } for layer in model.mlm_cls_list
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.enc_intermediate_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.enc_output_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.dec_intermediate_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.dec_output_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for layer in model.bert.encoder.layer
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in model.classifier_list.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ])
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.predictions.transform.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for layer in model.mlm_cls_list
        ])
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Metrics
    metric = evaluate.load("seqeval")
    accuracy_metric = evaluate.load("accuracy")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            ['B-'+label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            ['B-'+label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def get_generation_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone()
            y_true = references.detach().clone()
        else:
            y_pred = predictions.detach().cpu().clone()
            y_true = references.detach().cpu().clone()

        # Remove ignored index (special tokens)
        true_predictions = [
            [int(p) for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [int(l) for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        pred = []
        labels = []
        for l in true_predictions:
            pred.extend(l)
        for l in true_labels:
            labels.extend(l)
        return pred, labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        
    if args.only_do_predict:
        # print(model)
        # raise 'end'
        if args.prediction_tagging_file is not None:
            tagging_data_files["test"] = args.prediction_tagging_file
            extension = args.prediction_tagging_file.split(".")[-1]
            tagging_raw_datasets = load_dataset(extension, data_files=tagging_data_files)
            processed_tagging_raw_datasets = tagging_raw_datasets.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=tagging_raw_datasets["test"].column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            test_tagging_dataset = processed_tagging_raw_datasets["test"]
            test_tagging_dataloader = DataLoader(
                test_tagging_dataset, shuffle=False, collate_fn=tagging_data_collator, batch_size=args.per_device_train_batch_size
            )
            model, test_tagging_dataloader= accelerator.prepare(
                model, test_tagging_dataloader
            )

            model.eval()
            keep_id = label_to_id['$KEEP']
            import time
            start = time.time()
            for index, batch in enumerate(test_tagging_dataloader):
                batch['mode_head'] = torch.tensor(0).to(device)
                batch['task_num'] = torch.tensor(args.task_num).to(device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                probability = torch.nn.functional.softmax(outputs.logits, dim=-1)
                for pres, pros in zip(predictions, probability):
                    for _index, pre in enumerate(pres):
                        keep_pro = pros[_index][keep_id]
                        pre_pro = pros[_index][pre]
                        if keep_id != pre:
                            if pre_pro < args.min_pro:
                                # print('turn ', str(label_list[pre]), ' to keep.', str(pre_pro), ', ', str(keep_pro))
                                pres[_index] = keep_id
                            else:
                                # print('donnot turn ', str(label_list[pre]), ' to keep.', str(pre_pro), ', ', str(keep_pro))
                                pass
                labels = batch["labels"]
                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_tagging_dataloader) - 1:
                        predictions_gathered = predictions_gathered[: len(eval_tagging_dataloader.dataset) - samples_seen]
                        labels_gathered = labels_gathered[: len(eval_tagging_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += labels_gathered.shape[0]
                preds, refs = get_labels(predictions_gathered, labels_gathered)

                output_predictions_file = os.path.join(args.output_dir, "predictions.txt")
                with open(output_predictions_file, "a+") as writer:
                    for prediction in preds:
                        writer.write(" ".join(prediction) + "\n")
                metric.add_batch(
                    predictions=preds,
                    references=refs,
                )  # predictions and preferences are expected to be a nested list of labels, not label_ids
            end = time.time()
            print('test-time: ', end - start)
            eval_metric = compute_metrics()
            print(eval_metric)
        if args.prediction_generation_file is not None:
            generation_data_files["test"] = args.prediction_generation_file
            extension = args.prediction_generation_file.split(".")[-1]
            generation_raw_datasets = load_dataset(extension, data_files=generation_data_files)
            processed_generation_raw_datasets = generation_raw_datasets.map(
                tokenize_and_align_masks,
                batched=True,
                remove_columns=generation_raw_datasets["test"].column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            test_generation_dataset = processed_generation_raw_datasets["test"]
            test_generation_dataloader = DataLoader(
                test_generation_dataset, shuffle=False, collate_fn=generation_data_collator, batch_size=args.per_device_train_batch_size
            )
            model, test_generation_dataloader= accelerator.prepare(
                model, test_generation_dataloader
            )

            model.eval()
            results = []
            final_results = []
            import time
            start = time.time()
            for index, batch in enumerate(test_generation_dataloader):
                batch['mode_head'] = torch.tensor(1).to(device)
                batch['task_num'] = torch.tensor(args.task_num).to(device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_generation_dataloader) - 1:
                        predictions_gathered = predictions_gathered[: len(eval_generation_dataloader.dataset) - samples_seen]
                        labels_gathered = labels_gathered[: len(eval_generation_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += labels_gathered.shape[0]
                preds, refs = get_generation_labels(predictions_gathered, labels_gathered)
                results.append(preds)

                mask_index = 0
                for inputs in batch['input_ids']:
                    inputs = inputs.cpu().clone().numpy().tolist()
                    inputs_copy = []
                    in_delete = False
                    for index in range(len(inputs)):
                        if inputs[index] == tokenizer.convert_tokens_to_ids('[DELETE]') or inputs[index] == tokenizer.convert_tokens_to_ids('[TRANSFORM_VERB]'):
                            in_delete = True
                        elif inputs[index] == tokenizer.convert_tokens_to_ids('[/DELETE]') or inputs[index] == tokenizer.convert_tokens_to_ids('[/TRANSFORM_VERB]'):
                            in_delete = False
                        else:
                            if not in_delete:
                                inputs_copy.append(inputs[index])

                    for index in range(len(inputs_copy)):
                        if inputs_copy[index] == tokenizer.mask_token_id:
                            inputs_copy[index] = preds[mask_index]
                            mask_index += 1
                    final_results.append(inputs_copy)
                if len(preds) != mask_index:
                    raise ValueError("!!")
            end = time.time()
            print('test-time: ', end - start)
            output_predictions_file = os.path.join(args.output_dir, "final_results.txt")
            with open(output_predictions_file, "w") as writer:
                for prediction in final_results:
                    writer.write(tokenizer.decode(prediction, skip_special_tokens=True) + "\n")
        return

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    lenth_sum = 0
    if len(train_tagging_dataloader_list) > 0:
        for train_tagging_dataloader in train_tagging_dataloader_list:
            lenth_sum += len(train_tagging_dataloader)
    if len(train_generation_dataloader_list) > 0:
        for train_generation_dataloader in train_generation_dataloader_list:
            lenth_sum += len(train_generation_dataloader)
            
    num_update_steps_per_epoch = math.ceil(lenth_sum / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    datas = accelerator.prepare(
        model, optimizer, *train_tagging_dataloader_list, *eval_tagging_dataloader_list, *train_generation_dataloader_list, *eval_generation_dataloader_list, lr_scheduler
    )
    model, optimizer, train_tagging_dataloader_list, eval_tagging_dataloader_list, train_generation_dataloader_list, eval_generation_dataloader_list, lr_scheduler = \
        datas[0], datas[1], datas[2:2+len(train_tagging_dataloader_list)], datas[2+len(train_tagging_dataloader_list):2+len(train_tagging_dataloader_list)+len(eval_tagging_dataloader_list)], \
        datas[2+len(train_tagging_dataloader_list)+len(eval_tagging_dataloader_list):2+len(train_tagging_dataloader_list)+len(eval_tagging_dataloader_list)+len(train_generation_dataloader_list)], \
        datas[2+len(train_tagging_dataloader_list)+len(eval_tagging_dataloader_list)+len(train_generation_dataloader_list):2+len(train_tagging_dataloader_list)+len(eval_tagging_dataloader_list)+len(train_generation_dataloader_list)+len(eval_generation_dataloader_list)], \
        datas[-1]
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    lenth_sum = 0
    if len(train_tagging_dataloader_list) > 0:
        for train_tagging_dataloader in train_tagging_dataloader_list:
            lenth_sum += len(train_tagging_dataloader)
    if len(train_generation_dataloader_list) > 0:
        for train_generation_dataloader in train_generation_dataloader_list:
            lenth_sum += len(train_generation_dataloader)
    num_update_steps_per_epoch = math.ceil(lenth_sum / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("ner_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {lenth_sum}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // lenth_sum
            resume_step -= starting_epoch * lenth_sum

    # for mlm_cls in model.mlm_cls_list:
    #     mlm_cls.predictions.decoder.weight = model.get_input_embeddings().weight # decoder is not sparse
        # mlm_cls.predictions.decoder.weight = nn.Parameter(model.get_input_embeddings().weight.clone())
    device = accelerator.device
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        # for step, batch in enumerate(train_dataloader):
        for step in range(num_update_steps_per_epoch * args.gradient_accumulation_steps):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            if len(train_tagging_dataloader_list) == 0 or len(train_generation_dataloader_list) == 0:
                dataloader_sum = 1
                train_dataloaders_list = [[iter(train_tagging_dataloader) for train_tagging_dataloader in train_tagging_dataloader_list]] if len(train_tagging_dataloader_list) != 0 \
                    else [[iter(train_generation_dataloader) for train_generation_dataloader in train_generation_dataloader_list]]
                mode_head = [0] if len(train_tagging_dataloader_list) != 0 else [1]
            else:
                dataloader_sum = 2
                train_dataloaders_list = [[iter(train_tagging_dataloader) for train_tagging_dataloader in train_tagging_dataloader_list], [iter(train_generation_dataloader) for train_generation_dataloader in train_generation_dataloader_list]]
                mode_head = [0, 1]
            try:
                if len(args.copy_ids) == 1:
                    batch = next(train_dataloaders_list[step % dataloader_sum][0])
                else:
                    batch = next(train_dataloaders_list[step % dataloader_sum][int(step/dataloader_sum) % args.sum_tasks])
                # batch = next(train_dataloaders_list[step % dataloader_sum][2])

                # print(example)
            except StopIteration:
                train_dataloaders_list[step % dataloader_sum][int(step/dataloader_sum) % args.sum_tasks] = iter(train_dataloaders_list[step % dataloader_sum][int(step/dataloader_sum) % args.sum_tasks])
                if len(args.copy_ids) == 1:
                    batch = next(train_dataloaders_list[step % dataloader_sum][0])
                else:
                    batch = next(train_dataloaders_list[step % dataloader_sum][int(step/dataloader_sum) % args.sum_tasks])

            batch['mode_head'] = torch.tensor(mode_head[step % dataloader_sum]).to(device)
            batch['task_num'] = torch.tensor(int(step/dataloader_sum) % args.sum_tasks).to(device)
            # batch['task_num'] = torch.tensor(2).to(device)

            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % 10 == 0:
                print('loss: ', str(round(float(loss), 3)))
            if args.max_grad_norm != None:
                if accelerator.sync_gradients:
                    # print(1)
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type = args.norm_type)
            if step % args.gradient_accumulation_steps == 0 or step == lenth_sum - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        if len(train_tagging_dataloader_list) != 0 :
            for index, eval_tagging_dataloader in enumerate(eval_tagging_dataloader_list):
                for step, batch in enumerate(eval_tagging_dataloader):
                    batch['mode_head'] = torch.tensor(0).to(device)
                    batch['task_num'] = torch.tensor(index).to(device)

                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    labels = batch["labels"]
                    if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                    predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(eval_tagging_dataloader) - 1:
                            predictions_gathered = predictions_gathered[: len(eval_tagging_dataloader.dataset) - samples_seen]
                            labels_gathered = labels_gathered[: len(eval_tagging_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += labels_gathered.shape[0]
                    preds, refs = get_labels(predictions_gathered, labels_gathered)
                    metric.add_batch(
                        predictions=preds,
                        references=refs,
                    )  # predictions and preferences are expected to be a nested list of labels, not label_ids
            eval_metric = compute_metrics()
            accelerator.print(f"epoch {epoch}:", eval_metric)
            if args.with_tracking:
                accelerator.log(
                    {
                        "seqeval": eval_metric,
                        "train_loss": total_loss.item() / lenth_sum,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

        if len(train_generation_dataloader_list) != 0 :
            losses = []
            for index, eval_generation_dataloader in enumerate(eval_generation_dataloader_list):
                for step, batch in enumerate(eval_generation_dataloader):
                    batch['mode_head'] = torch.tensor(1).to(device)
                    batch['task_num'] = torch.tensor(index).to(device)

                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    predictions = outputs.logits.argmax(dim=-1)
                    labels = batch["labels"]
                    if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
                    predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
                    preds, refs = get_generation_labels(predictions_gathered, labels_gathered)
                    accuracy_metric.add_batch(
                        predictions=preds,
                        references=refs,
                    )
            results = accuracy_metric.compute()
            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy": results['accuracy'],
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss.item() / lenth_sum,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
        

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    from numpyencoder import NumpyEncoder
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
            
            all_results = {}
            if args.tagging_file_list is not None:
                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            if args.generation_file_list is not None:
                all_results.update({'perplexity': perplexity})
                all_results.update({'accuracy': results['accuracy']})
            if args.with_tracking:
                all_results.update({"train_loss": total_loss.item() / lenth_sum})
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f, cls=NumpyEncoder)


if __name__ == "__main__":
    main()