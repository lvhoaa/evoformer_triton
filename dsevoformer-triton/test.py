import torch 
from evoformer import EvoformerAttention

def max_neg_value(t):
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


def test_full_step(BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    res_mask = torch.randint(0, 2, (BATCH_SIZE, N_SEQ, 1, 1, SEQ_LEN), dtype=torch.bool, device="cuda") 
    pair_bias = (
        torch.empty(
            (BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN), dtype=torch.float32, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    ref_Q = Q.view((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM))
    ref_K = K.view((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM))
    ref_V = V.view((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM))
    ref_pair_bias = pair_bias.view((BATCH_SIZE, HEAD, 1, SEQ_LEN, SEQ_LEN))
    
    ref_P = (torch.matmul(ref_Q * softmax_scale, ref_K.transpose(3, 4)) + ref_pair_bias)
    ref_P = ref_P.masked_fill(~res_mask.view(BATCH_SIZE, 1, N_SEQ, 1, SEQ_LEN), max_neg_value(ref_P))
    ref_P = torch.softmax(ref_P.float(), dim=-1).to(dtype)
    ref_O = torch.matmul(ref_P, ref_V)
    ref_O = ref_O.reshape((BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM))
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_d_pair_bias, pair_bias.grad = pair_bias.grad.clone(), None

    # triton implementation
    tri_out = EvoformerAttention.apply(Q, K, V, res_mask, pair_bias).to(dtype)
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_d_pair_bias, pair_bias.grad = pair_bias.grad.clone(), None

    # compare
    rtol = 0.0 if dtype == torch.float16 else 1e-2 # allow error for bfloat16
    atol = 1e-2 
    
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_d_pair_bias, tri_d_pair_bias, atol=atol, rtol=rtol)
 
def tests_full_step(dtype):
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=128, DIM=64, dtype=dtype)    
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=128, DIM=32, dtype=dtype)    
    
    test_full_step(BATCH_SIZE=5, HEAD=10, N_SEQ=200, SEQ_LEN=384, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=5, HEAD=10, N_SEQ=200, SEQ_LEN=384, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=10, HEAD=5, N_SEQ=200, SEQ_LEN=384, DIM=64, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=10, N_SEQ=200, SEQ_LEN=640, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=10, N_SEQ=200, SEQ_LEN=640, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=15, N_SEQ=200, SEQ_LEN=640, DIM=64, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=100, SEQ_LEN=768, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=100, SEQ_LEN=768, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=10, N_SEQ=100, SEQ_LEN=768, DIM=64, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=1, N_SEQ=100, SEQ_LEN=1280, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=300, SEQ_LEN=512, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=300, SEQ_LEN=384, DIM=64, dtype=dtype)
    print("PASSED")

if __name__ == "__main__":
    tests_full_step(dtype=torch.bfloat16)
    # tests_full_step(dtype=torch.float16)