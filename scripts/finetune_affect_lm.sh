#!/bin/bash

DATA=ctrl_ft_v1
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRAD_ACCUM_STEPS=64

PYTHONPATH=. python scripts/finetune_affect_lm.py \
  --output_dir output/affect_doc_v1 \
	--model_type gpt2 \
  --model_name_or_path gpt2 \
  --do_train \
  --do_eval \
  --evaluate_during_training \
	--logging_steps 100 \
	--save_steps 100 \
  --per_gpu_train_batch_size=$BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
	--train_data_files $DATA/nontoxic.train.txt $DATA/toxic.train.txt \
	--eval_data_files $DATA/nontoxic.test.txt $DATA/toxic.test.txt \
  --learning_rate 0.001 \
  --freeze_transformer \
  --freeze_lm_head

# TODO: Try learning_rate between 1e-4 and 1e-2
