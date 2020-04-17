#!/bin/bash
mkdir -p output/affect
PYTHONPATH=. python scripts/finetune_affect_lm.py \
	--output_dir=output/affect/"$1" \
	--model_name_or_path=gpt2 \
	--do_train \
	--do_eval \
	--learning_rate=0.001 \
	--patience=2 \
	--num_train_epochs=20 \
	--evaluate_during_training \
	--log_after_epoch \
	--save_after_epoch \
	--per_gpu_train_batch_size=8 \
	--per_gpu_eval_batch_size=16 \
	--block_size=512 \
	--freeze_transformer \
	--freeze_lm_head

#	--gradient_accumulation_steps=8 \
# TODO: Try learning_rate between 1e-4 and 1e-2
