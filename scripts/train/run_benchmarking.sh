export WORLD_SIZE=???
export NODE_RANK=???
export MASTER_ADDR=???
export MASTER_PORT=12345

composer train.py \
    yamls/pretrain/examples/mpt-7b.yaml \
    max_seq_len=2048 \
    precision=amp_bf16 \
    data_local=/network/eldar/datasets/mpt_7b/c4 \
    train_loader.dataset.split=train_small \
    eval_loader.dataset.split=val_small \
    max_duration=100ba \
    eval_interval=1ep \
    device_eval_microbatch_size=6 \
    eval_first=False \
    scheduler.t_warmup=10ba \
    global_train_batch_size=192 \
    device_train_microbatch_size=6
