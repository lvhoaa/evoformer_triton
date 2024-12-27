import torch 
from deepspeed.utils.timer import SynchronizedWallClockTimer 
from evoformer import EvoformerAttention
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
from speed_benchmark import full_attention, full_deepspeed_evoformer_attention, full_evoformer_attention

device = 'cuda'
dtype = torch.bfloat16


BATCH, H, HEAD_DIM, N_SEQ = 4, 32, 64, 1
N_CTX = 384
provider = "triton"

if provider == "triton":
    q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True) # 4 * 1 * 384 * 32 * 64 * 2(bfloat16)
    k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device) 
    pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype, device=device)
    full_evoformer_attention(q, k, v, res_mask, pair_bias)
elif provider == "deepspeed":
    q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device).bfloat16() # deepspeed only works with bfloat16
    pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype, device=device)
    full_deepspeed_evoformer_attention(q, k, v, res_mask, pair_bias)    
elif provider == "torch":   
    q = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask = torch.randint(0, 2, (BATCH, 1, N_SEQ, 1, N_CTX), dtype=torch.bool, device=device)
    pair_bias = torch.randn((BATCH, H, 1, N_CTX, N_CTX), dtype=dtype, device=device)
    full_attention(q, k, v, res_mask, pair_bias)


mem_usage = SynchronizedWallClockTimer.memory_usage()
print("Memory usage after all: ", mem_usage)

# Memory benchmark full (msa row wise) -- max mem allocated
# deepspeed will have smaller mem bc of pair_bias bfloat16
# batch4-head32-dim64-nseq1
#     N_CTX  Triton [FP16]  deepspeed     torch
# 0   128.0       0.2502GB  0.0236GB   0.0686GB
# 1   256.0       0.2698GB  0.0549GB   0.1995GB
# 2   384.0       0.2972GB  0.0941GB   0.4085GB
# 3   512.0       0.3324GB  0.1411GB   0.6956GB
# 4   640.0       0.4694GB  0.1959GB   1.0608GB 
# 5   768.0       0.6570GB  0.2586GB   1.5042GB
# 6  1024.0       1.1260GB  0.4072GB   2.6253GB
# 7  2048.0       4.2520GB  1.3145GB  10.2346GB

# torch for 384: 
# qkv, output: 4 * 32 * 1 * 384 * 64 * 2 * 4 = 25,165,824
# res_mask, pair_bias: 4 * 1 * 1 * 384 * 1 + 4 * 32 * 384 * 384 * 2 = 37,750,272
# dot product + bias: 4 * 32 * 1 * 384 * 384 * (2+4) = 113,246,208
# grads for qkv, output, bias: 25,165,824 + 4 * 32 * 384 * 384 * 2 = 62,914,560

# triton for 384 
# qkv, output: 4 * 32 * 1 * 384 * 64 * 2 * 4 = 25,165,824
# res_mask, pair_bias: 4 * 1 * 1 * 384 * 1 + 4 * 32 * 384 * 384 * 2 = 37,750,272
# grads for qkv, output, bias: 62,914,560
# logsumexp: 4 * 32 * 1 * 384 * 4 = 196608
# D: 196608

# deepspeed
# qkv, output: 25,165,824
# res_mask, pair_bias: 4 * 1 * 1 * 384 * 2 + 4 * 32 * 384 * 384 * 2 = 37,751,808
# grads for qkv, output, bias: 62,914,560
# logsumexp: 4 * 32 * 1 * 384 * 4 = 196608
# D: 196608


# forward pass: 
# triton: 0.2972 GB 
# Memory usage after init:   | mem_allocated: 0.0527 GB | max_mem_allocated: 0.0527 GB | cache_allocated: 0.0566 GB | max_cache_allocated: 0.0566 GB
# Memory usage after init O and M:   | mem_allocated: 0.0588 GB | max_mem_allocated: 0.0588 GB | cache_allocated: 0.0762 GB | max_cache_allocated: 0.0762 GB
# Memory usage after fwd pass:   | mem_allocated: 0.0588 GB | max_mem_allocated: 0.2972 GB | cache_allocated: 0.3164 GB | max_cache_allocated: 0.3164 GB
# Memory usage after all:   | mem_allocated: 0.0588 GB | max_mem_allocated: 0.2972 GB | cache_allocated: 0.3164 GB | max_cache_allocated: 0.3164 GB

# deepspeed: 0.0588 GB
# Memory usage after init:   | mem_allocated: 0.0527 GB | max_mem_allocated: 0.0527 GB | cache_allocated: 0.0605 GB | max_cache_allocated: 0.0605 GB
# Memory usage after all:   | mem_allocated: 0.0588 GB | max_mem_allocated: 0.0588 GB | cache_allocated: 0.0605 GB | max_cache_allocated: 0.0605 GB