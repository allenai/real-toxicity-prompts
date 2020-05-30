#!/bin/bash

DATA=ctrl_ft_v1
BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=512

PYTHONPATH=. python scripts/finetune_ctrl.py \
	--output_dir output/ctrl_v1 \
	--model_type gpt2 \
	--model_name_or_path gpt2 \
	--do_train \
	--do_eval \
	--evaluate_during_training \
	--logging_steps 100 \
	--save_steps 100 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
	--train_data_files $DATA/nontoxic.train.txt $DATA/toxic.train.txt \
	--eval_data_files $DATA/nontoxic.test.txt $DATA/toxic.test.txt \
	--ctrl_codes "<|nontoxic|>" "<|toxic|>"
