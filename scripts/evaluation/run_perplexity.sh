# CONSTANT environment variables
SUBSAMPLES_DIR=/data/language-model-toxicity/data/webtext_subsamples
MODELS_DIR=/data/language-model-toxicity/models/gpt2-finetuned-models

WEBTEXT_TEST=$SUBSAMPLES_DIR/webtext_test_25_mil.npy
NONTOXIC_WEBTEXT_TEST=$SUBSAMPLES_DIR/webtext_test_non_toxic_25_mil.npy

BATCH_SIZE=4

#############################
# Affect
#############################
MODEL_PATH=$MODELS_DIR/affect_doc_v1
CONTROLLABLE=affect-gpt2

# 1. Affect on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect_gpt2_eval
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --controllable_model $CONTROLLABLE \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR

# 2. Affect on NONTOXIC
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/affect_gpt2_nontoxic_eval
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --controllable_model $CONTROLLABLE \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR

############################
# CTRL-GPT2
############################
MODEL_PATH=$MODELS_DIR/gpt2_ctrl_v3_random_sample
CONTROLLABLE=ctrl-gpt2

# 3. CTRL-GPT2 on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl_gpt2_eval
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --controllable_model $CONTROLLABLE \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR

# 4. CTRL-GPT2 on NONTOXIC
EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/ctrl_gpt2_nontoxic_eval
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --controllable_model $CONTROLLABLE \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR

#############################
# DAPT-NONTOXIC
#############################
MODEL_PATH=$MODELS_DIR/finetune_lte2_v2

# 5. DAPT-NONTOXIC-GPT2 on FULL DATA
EVAL_FILE=$WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_dapt_lte2_eval
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR

EVAL_FILE=$NONTOXIC_WEBTEXT_TEST
OUTPUT_DIR=output/eval/gpt2_dapt_lte2_nontoxic_eval
python -m scripts.evaluation.run_language_modeling_webtext \
  --model_type gpt2 \
  --model_name_or_path $MODEL_PATH \
  --do_eval \
  --webtext \
  --eval_data_file $EVAL_FILE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR
