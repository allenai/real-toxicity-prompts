GEN_SAMPLES=10000

#python -m scripts.run_prompts_experiments \
#	--model_type gpt2-affect \
#	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/affect_lm_doc_v1 \
#	--gen_batch_size 64 \
#	--gen_samples $GEN_SAMPLES \
#	--eos_prompt \
#	output/prompts/eos/gpt2_affect_eos

#python -m scripts.run_prompts_experiments \
#	--model_type ctrl \
#	--model_name_or_path ctrl \
#	--gen_batch_size 64 \
#	--gen_samples $GEN_SAMPLES \
#	--eos_prompt \
#	output/prompts/eos/ctrl_eos
#
#python -m scripts.run_prompts_experiments \
#	--model_type gpt2-ctrl \
#	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/gpt2_ctrl_v3_random_sample \
#	--gen_batch_size 64 \
#	--gen_samples $GEN_SAMPLES \
#	--eos_prompt \
#	output/prompts/eos/gpt2_ctrl_eos

#python -m scripts.run_prompts_experiments \
#	--model_type gpt2-affect \
#	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/affect_lm_doc_v1 \
#	--gen_batch_size 64 \
#	--gen_samples $GEN_SAMPLES \
#	--eos_prompt \
#	output/prompts/eos/gpt2_affect_beta1_eos
#
#python -m scripts.run_prompts_experiments \
#	--model_type gpt2 \
#	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/finetune_lte2_v2 \
#	--gen_batch_size 64 \
#	--gen_samples $GEN_SAMPLES \
#	--eos_prompt \
#	output/prompts/eos/gpt2_lte2_eos
#
#python -m scripts.run_prompts_experiments \
#	--model_type gpt2 \
#	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/finetune_gte99_v2 \
#	--gen_batch_size 64 \
#	--gen_samples $GEN_SAMPLES \
#	--eos_prompt \
#	output/prompts/eos/gpt2_gte99_eos

#python -m scripts.run_prompts_experiments \
#	--model_type pplm \
#	--model_name_or_path pplm \
#	--gen_batch_size 8 \
#	--gen_samples 1000 \
#	--eos_prompt \
#	output/prompts/eos/pplm_eos

python -m scripts.run_prompts_experiments \
  --model_type gpt2 \
  --model_name_or_path gpt2 \
  --gen_batch_size 64 \
  --gen_samples $GEN_SAMPLES \
  --eos_prompt \
  output/prompts/eos/gpt2_eos
