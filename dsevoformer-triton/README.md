# EvoformerAttention in Triton 
Original DS4Sci_EvoformerAttention is in https://arxiv.org/abs/2310.04610, https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/, https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/ops/deepspeed4science/evoformer_attn.py

# Kernel requirements
- Work with AMD hardware
- The crop size used in training is 384, but there exists inputs that are shorter than crop size. So ideally, the kernel should work with all sequence lengths (SEQ_LEN)
- Works with all head dimensions HEAD_DIM (32, 64, etc.)
- Good speed & memory optimizations

## Current progress of Triton EvoformerAttention: 
I think the EvoformerAttention kernel inside evoformer.py is good in terms of correctness. It is unit-tested (run tests in test.py) and loss-tested (similar loss to pytorch implementation, in real data training)

But there are a couple of problems: 


### Problem 1: Performance issue: Speed 
In speed_benchmark.py, I include some profiled results at the end. The forward pass is good, even better than DS4Sci_EvoformerAttention. However, the backward pass is very slow, slower than DS4Sci_EvoformerAttention and normal pytorch implementation. So the overall fwd+bwd speed is similar to pytorch implementation.

I did some analysis to find out that in the backward pass, the tl.atomic_add is what causes slowdown. So I think tl.atomic_add needs a review.

*Update*: I have tuned/reduced the block size in autotune configs and BLOCK_SIZE_MACRO & BLOCK_SIZE_MICRO so now the speed is better. But because Triton EvoformerAttention speed is still less than DS4Sci_EvoformerAttention, I think there will still be room for improvements. 


### Problem 2: Performance issue: Memory
In memory_benchmark.py, I include profiled results at the end. Current Triton EvoformerAttention is just slightly better than PyTorch implementation. 

I think current Triton EvoformerAttention implementation just mimics Flash-attention in tiling operations. I just allocate HBM memory for input and output in the code (via torch.empty()). All operations are loaded and done in SRAM. However, the max_memory_allocated rises from 0.0588 GB to 0.2972 GB after running the kernel. So, I guess the problem is register spilling (see ptxas.log). 

I have tried reducing the block size, but it didn't improve the results. 


### Problem 3: The current kernel only works with SEQ_LEN % 128 == 0
*Update: I have fixed this one by mask-loading and mask-storing.* 
<!-- 
This is because I worked with the original Flash-attention triton code on https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html, which only allows SEQ_LEN % 128==0. 

I also found other flash-attn triton implementations but I don't think they fit our requirements: 
+ https://github.com/triton-lang/triton/blob/c1166e537552e73392d26b055ea1b505277ca242/python/test/unit/hopper/test_flashattention.py: this one is for Hopper architecture 
+ https://github.com/triton-lang/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py: this one only works with HEAD_DIM=64, and I don't see any AMD/HIP related info
+ https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py: this one is an experimental implementation that starts from the one above and only works with HEAD_DIM=64

That is why I decided to go with https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html. 

I will try to work around by implementing mask-loading to make it fit all sequence lengths.  -->