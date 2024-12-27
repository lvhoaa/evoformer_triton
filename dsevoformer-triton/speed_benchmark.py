import torch
import triton
import triton.language as tl
from evoformer import EvoformerAttention
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def _attention(q, k, v, res_mask, pair_bias):
    softmax_scale = 1 / (q.shape[-1]**0.5)
    ref_P = (torch.matmul(q * softmax_scale, k.transpose(3, 4)) + pair_bias)
    ref_P = ref_P.masked_fill(~res_mask, max_neg_value(ref_P)) 
    ref_P = torch.softmax(ref_P.float(), dim=-1).to(q.dtype)
    ref_O = torch.matmul(ref_P, v)
    return ref_O
    
def full_evoformer_attention(q, k, v, res_mask, pair_bias):
    o = EvoformerAttention.apply(q, k, v, res_mask, pair_bias)
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)

def full_deepspeed_evoformer_attention(q, k, v, res_mask, pair_bias):
    o = DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)

def full_attention(q, k, v, res_mask, pair_bias):
    BATCH, H, N_SEQ, N_CTX, HEAD_DIM = q.shape
    o = _attention(q, k, v, res_mask, pair_bias)
    o= o.reshape((BATCH, N_SEQ, N_CTX, H, HEAD_DIM))
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)

TORCH_HAS_FP8 = False
HAS_FLASH=False
BATCH, N_HEADS, HEAD_DIM, N_SEQ = 4, 32, 64, 1 
configs = []
for mode in ["fwd", "bwd", "full"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[128, 256, 384, 512, 640, 768, 1024, 2048],
            line_arg="provider",
            line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) + (["deepspeed"]) + (["torch"]),
            line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) + (["deepspeed"]) + (["torch"]),
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"evoformer-attention-batch{BATCH}-head{N_HEADS}-dim{HEAD_DIM}-nseq{N_SEQ}-{mode}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "N_SEQ": N_SEQ,
                "mode": mode,
            },
        ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, N_SEQ, mode, provider, device='cuda'):
    assert mode in ["fwd", "bwd", "full"]
    dtype = torch.bfloat16
    if "triton" in provider:
        q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device) 
        pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=torch.float32, device=device)
        fn = lambda: EvoformerAttention.apply(q, k, v, res_mask, pair_bias)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_evoformer_attention(q, k, v, res_mask, pair_bias)
        ms = triton.testing.do_bench(fn)
        
    if "deepspeed" in provider: 
        q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device).bfloat16() # deepspeed only works with bfloat16
        pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype, device=device)
        fn = lambda: DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_deepspeed_evoformer_attention(q, k, v, res_mask, pair_bias)
        ms = triton.testing.do_bench(fn)
        
    if provider == "torch":
        q = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        res_mask = torch.randint(0, 2, (BATCH, 1, N_SEQ, 1, N_CTX), dtype=torch.bool, device=device)
        pair_bias = torch.randn((BATCH, H, 1, N_CTX, N_CTX), dtype=torch.float32, device=device)

        fn = lambda: _attention(q, k, v, res_mask, pair_bias)
        if mode == "bwd":
            o = fn()
            o= o.reshape((BATCH, N_SEQ, N_CTX, H, HEAD_DIM))
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_attention(q, k, v, res_mask, pair_bias)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    if mode == "full":
        total_flops *= 3.5 # 1.0 (forward) + 2.5 (backward)
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    bench_flash_attention.run(save_path=".", print_data=True)



# TFLOPS all 3:
# evoformer-attention-batch4-head32-dim64-nseq1-fwd:
#     N_CTX  Triton [FP16]  deepspeed     torch
# 0   128.0      18.276691  14.149612  5.017307
# 1   256.0      34.366910  20.449832  6.130828
# 2   384.0      45.508338  28.778877  6.673687
# 3   512.0      51.147998  30.298855  6.992558
# 4   640.0      59.609685  35.281573  7.095920
# 5   768.0      61.204355  34.508447  7.209146
# 6  1024.0      67.893515  36.595078  7.295552
# 7  2048.0      74.555261  38.811609  6.663162
# evoformer-attention-batch4-head32-dim64-nseq1-bwd:
#     N_CTX  Triton [FP16]  deepspeed      torch
# 0   128.0       4.855558  14.349185   7.809936
# 1   256.0       5.988150  18.393406  10.369143
# 2   384.0       6.860687  18.626663  11.444663
# 3   512.0       7.300234  19.161620  12.252747
# 4   640.0       7.562927  19.871882  12.519841
# 5   768.0       7.255784  20.080914  12.827279
# 6  1024.0       7.539474  19.880361  13.120965
# 7  2048.0       7.842936  20.185059  13.578925
# evoformer-attention-batch4-head32-dim64-nseq1-full:
#     N_CTX  Triton [FP16]  deepspeed      torch
# 0   128.0       5.105708  11.752278   4.667564
# 1   256.0       7.772714  18.812647   8.714177
# 2   384.0       8.996056  20.595664   9.507706
# 3   512.0       9.616603  21.289168  10.056617
# 4   640.0      10.028462  22.557961  10.253296
# 5   768.0       9.660762  22.667459  10.458976
# 6  1024.0      10.076945  22.725293  10.655825
# 7  2048.0      10.520398  23.328883  10.457865


# Further Analysis
# without tl.atomic_add: speed is much faster 
# fused-attention-batch4-head32-d64-bwd:
#     N_CTX  Triton [FP16]      torch
# 0   128.0       7.795099   7.736862
# 1   256.0      17.705724  10.453049
# 2   384.0      22.118004  11.476177
# 3   512.0      25.020863  12.290309
# 4   640.0      26.644472  12.476801
# 5   768.0      26.012670  12.860703
# 6  1024.0      28.317055  13.161550
# 7  2048.0      30.260229  13.592310
