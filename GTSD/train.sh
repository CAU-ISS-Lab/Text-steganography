python trainer_main.py \
model.name='bert-base-uncased' batch_size=128 grad_accum=1 \
total_steps=40000 exp.name=movie \
data.path='./data' num_workers=0 \
self_condition=False \
data.name=commongen tgt_len=54 max_pos_len=128 lr=3e-4 lr_step=40000 \
intermediate_size=512 num_attention_heads=8 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \
