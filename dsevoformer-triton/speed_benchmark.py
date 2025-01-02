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
            x_vals=[32, 64, 128, 200, 256, 300, 384, 400, 512, 600, 640, 700, 768, 1024, 2048],
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
#   evoformer-attention-batch4-head32-dim64-nseq1-fwd:
#      N_CTX  Triton [FP16]  deepspeed     torch
# 0     32.0       1.687417   1.662721  0.546744
# 1     64.0       6.470241   6.307206  2.041048
# 2    128.0      19.035393  14.125626  5.031370
# 3    200.0      19.443039  13.409253  5.810879
# 4    256.0      34.136464  20.341535  6.116298
# 5    300.0      28.085248  21.325721  5.524494
# 6    384.0      42.738891  28.836533  6.690574
# 7    400.0      42.931564  22.913086  6.558062
# 8    512.0      50.843154  30.375329  7.005505
# 9    600.0      39.494768  31.333343  7.005906
# 10   640.0      58.712700  35.189433  7.105537
# 11   700.0      39.754620  31.899943  6.280933
# 12   768.0      59.103375  34.443182  7.220005
# 13  1024.0      63.772784  36.585324  7.299122
# 14  2048.0      73.069228  38.787800  6.662004
#  evoformer-attention-batch4-head32-dim64-nseq1-bwd:
#      N_CTX  Triton [FP16]  deepspeed      torch
# 0     32.0       0.502407   2.111835   0.489542
# 1     64.0       2.081131   7.479326   1.969577
# 2    128.0       8.416285  14.369970   7.882652
# 3    200.0       8.115167  11.677120  10.054638
# 4    256.0      10.708642  18.212603  10.393641
# 5    300.0      10.029241  15.880150   9.574828
# 6    384.0      11.666462  18.981025  11.435212
# 7    400.0      11.132026  15.437630  11.471562
# 8    512.0      12.258256  19.464519  12.253987
# 9    600.0      11.142554  17.080876  12.402142
# 10   640.0      12.501690  19.560781  12.537632
# 11   700.0      11.327843  18.887332  11.403697
# 12   768.0      12.676891  19.999608  12.833978
# 13  1024.0      12.956665  20.013812  13.119499
# 14  2048.0      13.381272  20.407260  13.583875
# evoformer-attention-batch4-head32-dim64-nseq1-full:
#      N_CTX  Triton [FP16]  deepspeed      torch
# 0     32.0       0.318134   0.746090   0.292363
# 1     64.0       1.268601   2.991435   1.165519
# 2    128.0       5.118879  11.766904   4.625459
# 3    200.0       9.724850  12.197692   8.324912
# 4    256.0      13.150785  18.683410   8.707463
# 5    300.0      12.155901  17.001647   7.900240
# 6    384.0      14.568522  20.889341   9.503093
# 7    400.0      13.985161  16.887936   9.428200
# 8    512.0      15.518561  21.579404  10.064401
# 9    600.0      13.925444  19.503459  10.122138
# 10   640.0      16.007398  22.260215  10.258208
# 11   700.0      14.153905  21.249433   9.215166
# 12   768.0      16.250124  22.577245  10.468811
# 13  1024.0      16.691375  22.821708  10.655583
# 14  2048.0      17.423978  23.519266  10.460426



# Further Analysis
# without tl.atomic_add: speed is much faster 
# evoformer-attention-batch4-head32-d64-bwd:
#     N_CTX  Triton [FP16]      torch
# 0   128.0       7.795099   7.736862
# 1   256.0      17.705724  10.453049
# 2   384.0      22.118004  11.476177
# 3   512.0      25.020863  12.290309
# 4   640.0      26.644472  12.476801
# 5   768.0      26.012670  12.860703
# 6  1024.0      28.317055  13.161550
# 7  2048.0      30.260229  13.592310
