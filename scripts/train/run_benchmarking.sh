#export WORLD_SIZE=32
#export NODE_RANK=3
#export MASTER_ADDR=192.168.201.158
#export MASTER_PORT=12348

composer train.py \
    yamls/pretrain/examples/mpt-7b.yaml \
    max_seq_len=1024 \
    precision=amp_bf16 \
    data_local=/network/eldar/mpt_7b/c4_1024 \
    train_loader.dataset.split=train_small \
    eval_loader.dataset.split=val_small \
    max_duration=100ba \
    eval_interval=1ep \
    eval_first=False \
    model.attn_config.attn_impl=flash \
    fsdp_config.activation_checkpointing=True \
    scheduler.t_warmup=10ba \
    global_train_batch_size=1024 \
    device_train_microbatch_size=64 \
    device_eval_microbatch_size=64
