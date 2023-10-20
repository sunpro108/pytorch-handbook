## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp.py
CUDA_VISIBLE_DEVICES="0" \
torchrun --standalone \
--nnodes=1 \
--nproc-per-node=1 \
main.py
