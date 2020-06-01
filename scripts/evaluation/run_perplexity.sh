# SETUP
BATCH_SIZE=4

run_perplexity() {
  if [ -z "$4" ]; then
    echo "Evaluating pretrained model: $1"
    echo "model: $1, data: $2, output: $3"
    python -m scripts.evaluation.run_language_modeling_webtext \
      --model_type gpt2 \
      --model_name_or_path "$1" \
      --do_eval \
      --webtext \
      --eval_data_file "$2" \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --output_dir "$3"
  else
    echo "Evaluating controllable model: $4"
    echo "model: $1, data: $2, output: $3"
    python -m scripts.evaluation.run_language_modeling_webtext \
      --model_type gpt2 \
      --model_name_or_path "$1" \
      --do_eval \
      --webtext \
      --eval_data_file "$2" \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --output_dir "$3" \
      --controllable_model "$4"
  fi
}

SUBSAMPLES_DIR=/data/language-model-toxicity/data/webtext_subsamples
MODELS_DIR=/data/language-model-toxicity/models/gpt2-finetuned-models

WEBTEXT_TEST=$SUBSAMPLES_DIR/webtext_test_25_mil.npy
NONTOXIC_WEBTEXT_TEST=$SUBSAMPLES_DIR/webtext_test_non_toxic_25_mil.npy

#############################
# Affect
#############################
MODEL_PATH=$MODELS_DIR/affect_lm_doc_v1
CONTROLLABLE=affect-gpt2

# 1. Affect on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect_gpt2_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

# 2. Affect on NONTOXIC
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect_gpt2_nontoxic_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

############################
# CTRL-GPT2
############################
MODEL_PATH=$MODELS_DIR/gpt2_ctrl_v3_random_sample
CONTROLLABLE=ctrl-gpt2

# 3. CTRL-GPT2 on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl_gpt2_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

# 4. CTRL-GPT2 on NONTOXIC
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl_gpt2_nontoxic_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

#############################
# DAPT-NONTOXIC
#############################
MODEL_PATH=$MODELS_DIR/finetune_lte2_v2
CONTROLLABLE=""

# 5. DAPT-NONTOXIC-GPT2 on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_dapt_lte2_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

# 6. DAPT-NONTOXIC-GPT2 on NONTOXIC
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_dapt_lte2_nontoxic_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

#############################
# DAPT-NONTOXIC
#############################
MODEL_PATH=$MODELS_DIR/finetune_gte99_v2
CONTROLLABLE=""

# 5. DAPT-TOXIC-GPT2 on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_dapt_gte99_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

# 6. DAPT-TOXIC-GPT2 on NONTOXIC
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_dapt_gte99_nontoxic_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

#############################
# GPT-2
#############################
MODEL_PATH=gpt2
CONTROLLABLE=""

# 8. GPT-2 on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE

# 9. GPT-2 on FULL DATA
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_nontoxic_eval
run_perplexity $MODEL_PATH $EVAL_FILE $OUTPUT_DIR $CONTROLLABLE
