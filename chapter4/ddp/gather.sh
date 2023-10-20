CUDA_VISIBLE_DEVICES="0,1" \
torchrun --standalone \
--nnodes=1 \
--nproc-per-node=2 \
gather.py