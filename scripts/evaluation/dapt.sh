# CONSTANT environment variables
WEBTEXT_TEST=x19_tokens.npy
NONTOXIC_WEBTEXT_TEST=webtext_test_non_toxic.npy
BATCH_SIZE=32
AFFECT_MODEL_PATH=affect_doc_v1
CTRL_MODEL_PATH=gpt2_ctrl_v3_random_sample

# 1. Affect on FULL DATA
MODEL_PATH=$AFFECT_MODEL_PATH
CONTROLLABLE=affect-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect-gpt2-nontoxic

# 2. Affect on NONTOXIC
MODEL_PATH=$AFFECT_MODEL_PATH
CONTROLLABLE=affect-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect-gpt2-nontoxic

# 3. CTRL-GPT2 on FULL DATA
MODEL_PATH=$CTRL_MODEL_PATH
CONTROLLABLE=ctrl-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl-gpt2

# 4. CTRL-GPT2 on NONTOXIC
MODEL_PATH=$CTRL_MODEL_PATH
CONTROLLABLE=ctrl-gpt2
BATCH_SIZE=$BATCH_SIZE
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl-gpt2-nontoxic

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
