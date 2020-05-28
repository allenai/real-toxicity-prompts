# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import logging
import math
import os
from dataclasses import dataclass, field
from itertools import groupby
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def split_webtext_shard(shard: np.array, block_size: int, eos: int) -> List[np.array]:
    idx = np.nonzero(shard == eos)[0] + 1  # Get start indices of each document split
    idx = idx[:-1]  # Last split is empty, so remove it
    docs = np.split(shard, idx)[1:]  # Split and discard EOS at start
    print("WebText: Loaded", len(docs), "documents from shard")

    examples = []
    for tokens in docs:
        for i in range(0, len(tokens), block_size):
            examples.append(tokens[i: i + block_size])

    return examples


def make_bucket_batches(examples: List[np.array], batch_size: int):
    batches = []
    for bucket_len, bucket in groupby(examples, len):
        bucket = list(bucket)
        for i in range(0, len(bucket), batch_size):
            batches.append(bucket[i: i + batch_size])
    return batches


class WebTextLineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, max_batch_size: int,
                 local_rank=-1):
        assert os.path.isfile(file_path)
        logger.info(f"WebText: Loading WebText test set features from {file_path}, block size {block_size}")

        shard = np.load(file_path)
        examples = split_webtext_shard(shard, block_size=block_size, eos=tokenizer.eos_token_id)
        logger.info(f"WebText: Loaded {len(examples)} blocks")

        # Create batches of equal length
        self.batches = make_bucket_batches(examples, max_batch_size)
        logger.info(f"WebText: Created {len(self.batches)} batches with max size {max_batch_size}")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i) -> List[torch.Tensor]:
        return [torch.tensor(x, dtype=torch.long) for x in self.batches[i]]


class BucketEvalTrainer(Trainer):
    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 prediction_loss_only: Optional[bool] = None) -> Dict[str, float]:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Set batch size to None to disable automatic batching
        eval_dataloader = DataLoader(eval_dataset if eval_dataset is not None else self.eval_dataset,
                                     batch_size=None,
                                     shuffle=False,
                                     collate_fn=self.data_collator.collate_batch)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")
        return output.metrics


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    webtext: bool = field(
        default=False,
        metadata={"help": "Use webtext eval set"},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, batch_size: int, evaluate=False,
                local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.webtext:
        return WebTextLineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, max_batch_size=batch_size,
            local_rank=local_rank
        )
    elif args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank
        )
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank,
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, batch_size=training_args.train_batch_size,
                    local_rank=training_args.local_rank)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, batch_size=training_args.eval_batch_size,
                    local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = BucketEvalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()
