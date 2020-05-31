# CONSTANT environment variables
SUBSAMPLES_DIR=/data/language-model-toxicity/data/webtext_subsamples
MODELS_DIR=/data/language-model-toxicity/models/gpt2-finetuned-models

WEBTEXT_TEST=$SUBSAMPLES_DIR/webtext_test_25_mil.npy
NONTOXIC_WEBTEXT_TEST=$SUBSAMPLES_DIR/webtext_test_non_toxic_25_mil.npy

BATCH_SIZE=4

AFFECT_MODEL_PATH=$MODELS_DIR/affect_doc_v1
CTRL_MODEL_PATH=$MODELS_DIR/gpt2_ctrl_v3_random_sample
DAPT_NONTOXIC_PATH=$MODELS_DIR/finetune_lte2_v2
DAPT_TOXIC_PATH=$MODELS_DIR/finetune_gte99_v2

# 1. Affect on FULL DATA
MODEL_PATH=$AFFECT_MODEL_PATH
CONTROLLABLE=affect-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect_gpt2_eval

# 2. Affect on NONTOXIC
MODEL_PATH=$AFFECT_MODEL_PATH
CONTROLLABLE=affect-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect_gpt2_nontoxic_eval

# 3. CTRL-GPT2 on FULL DATA
MODEL_PATH=$CTRL_MODEL_PATH
CONTROLLABLE=ctrl-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl_gpt2_eval

# 4. CTRL-GPT2 on NONTOXIC
MODEL_PATH=$CTRL_MODEL_PATH
CONTROLLABLE=ctrl-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl_gpt2_nontoxic_eval

# Run this command with different environment variables
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --controllable_model $CONTROLLABLE \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR
