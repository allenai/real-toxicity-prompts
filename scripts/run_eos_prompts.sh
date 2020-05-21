GEN_SAMPLES=10000

python -m scripts.run_prompts_experiments \
	--model_type gpt2-affect \
	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/affect_lm_doc_v1 \
	--gen_batch_size 64 \
	--gen_samples $GEN_SAMPLES \
	--eos_prompt \
	output/prompts/eos/gpt2_affect_eos

python -m scripts.run_prompts_experiments \
	--model_type ctrl \
	--model_name_or_path ctrl \
	--gen_batch_size 64 \
	--gen_samples $GEN_SAMPLES \
	--eos_prompt \
	output/prompts/eos/ctrl_eos

python -m scripts.run_prompts_experiments \
	--model_type gpt2-ctrl \
	--model_name_or_path /data/language-model-toxicity/models/gpt2-finetuned-models/gpt2_ctrl_v3_random_sample \
	--gen_batch_size 64 \
	--gen_samples $GEN_SAMPLES \
	--eos_prompt \
	output/prompts/eos/gpt2_ctrl_eos
