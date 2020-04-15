#!/bin/bash
mkdir -p output/affect
PYTHONPATH=. python scripts/finetune_affect_lm.py \
	--output_dir=output/affect/"$1" \
	--model_name_or_path=gpt2 \
	--do_train \
	--do_eval \
	--num_train_epochs=20 \
	--evaluate_during_training \
	--log_after_epoch \
	--save_after_epoch \
	--per_gpu_train_batch_size=2 \
	--per_gpu_eval_batch_size=8 \
	--gradient_accumulation_steps=8 \
	--block_size=512 \
	--affect_beta=5 \
	--freeze_transformer \
	--freeze_lm_head
