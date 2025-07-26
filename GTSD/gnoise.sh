#！/bin/bash

python gennoise.py \
	model.name='./bert-base-uncased' batch_size=256 \
	exp.name=commongen load_step=120000 \
	num_workers=0 \
	load_from_ema=True \
	self_condition=False \
	pred_len=False \
	pred_len_strategy='token_embed' \
	data.path='./data' data.name=movie max_pos_len=128 num_samples=13 \
	intermediate_size=512 num_attention_heads=8 \
	in_channels=64 out_channels=64 time_channels=64 \
	skip_sample=True gen_timesteps=20 \
	schedule_sampler='xy_uniform' time_att=True att_strategy='null' \
	tgt_len=54 prediction=True seed=42 numm=10
#numm: the value controls the parameters of original noises
