import os
import torch
import torch.distributed as dist

local_rank = int(os.environ['LOCAL_RANK'])
# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# All tensors below are of torch.int64 dtype.
# We have 3 process groups, 2 ranks.
num_rank = 2
# tensor_list = [torch.zeros(3, dtype=torch.int64).to(local_rank) for _ in range(num_rank)]
# print('tensor_list',tensor_list)
# tensor = torch.arange(3, dtype=torch.int64) + 1 + 2 * local_rank
# tensor = tensor.to(local_rank)
# print('tensor',tensor)
# dist.all_gather(tensor_list, tensor)
# print(tensor_list)


# # All tensors below are of torch.cfloat dtype.
# # We have 3 process groups, 2 ranks.
# tensor_list = [torch.zeros(3, dtype=torch.cfloat).to(local_rank) for _ in range(num_rank)]
# print(tensor_list)
# tensor = torch.tensor([1+1j, 2+2j, 3+3j], dtype=torch.cfloat) + 2 * local_rank * (1+1j)
# tensor = tensor.to(local_rank)
# dist.all_gather(tensor_list, tensor)
# print(tensor_list)


####
# Note: Process group initialization omitted on each rank.
# Assumes world_size of 3.
gather_objects = [f'str@{local_rank}',f'str2@{local_rank}',f'str3@{local_rank}'] # any picklable object
output = [None for _ in range(2)]
dist.all_gather_object(output, gather_objects)
output = [item for sublist in output for item in sublist]
print(output)