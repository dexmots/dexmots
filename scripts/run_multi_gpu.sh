
torchrun  \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_shac.py --cfg cfg/shac/allegro.yaml \
    --logdir logs/Allegro/test --no-time-stamp --multi-gpu
