OUTPUT_DIR=output
FINETUNE_DIR=$OUTPUT_DIR/finetuned_models
PROMPTS_DIR=$OUTPUT_DIR/prompts
mkdir -p $OUTPUT_DIR $FINETUNE_DIR $PROMPTS_DIR $PROMPTS_DIR/experiments $PROMPTS_DIR/datasets

# Download prompts dataset
gsutil cp gs://lm-toxicity-data/prompts_n_50percent.pkl $PROMPTS_DIR/datasets

# Download fine-tuned models
LTE_2_DIR=$FINETUNE_DIR/finetune_toxicity_percentile_lte2/finetune_output
GTE_99_DIR=$FINETUNE_DIR/finetune_toxicity_percentile_gte99/finetune_output
MIDDLE_20_DIR=$FINETUNE_DIR/finetune_toxicity_percentile_middle_20_subsample/finetune_output
mkdir -p $LTE_2_DIR $GTE_99_DIR $MIDDLE_20_DIR

gsutil cp gs://lm-toxicity-data/finetune_toxicity_percentile_lte2.tar.gz $LTE_2_DIR
cd $LTE_2_DIR && tar -xf finetune_toxicity_percentile_lte2.tar.gz && cd -

gsutil cp gs://lm-toxicity-data/finetune_toxicity_percentile_gte99.tar.gz $GTE_99_DIR
cd $GTE_99_DIR && tar -xf finetune_toxicity_percentile_gte99.tar.gz && cd -

gsutil cp gs://lm-toxicity-data/finetune_toxicity_percentile_middle_20_subsample.tar.gz $MIDDLE_20_DIR
cd $MIDDLE_20_DIR && tar -xf finetune_toxicity_percentile_middle_20_subsample.tar.gz && cd -
