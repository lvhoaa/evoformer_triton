# NOTE: only works with SEQ_LEN that is multiple of 128.
# NOTE: investigate poor performance issue vs standard pytorch: time & memory 
# TODO:  In our kernels, we tune the tile size for better performance. Large tile size leads to more efficient memory access while incurring register spilling; We tune the tile size to be (64, 64, 1)
# TODO: To reduce the contention that multiple thread blocks are trying to write the same place, we schedule the thread block so that blocks executing on GPU’s multiprocessors at the same wave write to different tiles.

import torch
import triton
import triton.language as tl
from deepspeed.utils.timer import SynchronizedWallClockTimer 

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def get_neg_max_value(dtype: tl.dtype):
    if dtype == tl.float32:
        return -3.4028234663852886e+38
    if dtype == tl.float16:
        return -65504.0
    if dtype == tl.bfloat16:
        return -3.3895313892515355e+38

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    res_mask_block_ptr,
    pair_bias_block_ptr,
    block_index_q,
    stride_mask_seq,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    Q_block = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        K_block = tl.load(K_block_ptr)
        pair_bias_block = tl.load(pair_bias_block_ptr)
        res_mask_block = tl.load(res_mask_block_ptr).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
        
        QK_block = tl.dot(Q_block, K_block) + pair_bias_block
        QK_block = tl.where(res_mask_block, QK_block, get_neg_max_value(QK_block.dtype))
        
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block = QK_block - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(V_block.dtype)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        pair_bias_block_ptr = tl.advance(pair_bias_block_ptr, (0, BLOCK_SIZE_KV))
        res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq
        
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([1] if is_hip() else [3, 4, 7])
        for num_warps in [4, 8]
    ],
    key=["SEQ_LEN", "DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM
    K,  # BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM
    V,  # BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM
    res_mask, # BATCH_SIZE, 1, N_SEQ, SEQ_LEN, 1
    pair_bias, # BATCH_SIZE, HEAD, 1, SEQ_LEN, SEQ_LEN
    softmax_scale,
    M,  # BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN
    O,  # BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_msa,
    stride_Q_seq,
    stride_Q_dim,
    
    stride_K_batch,
    stride_K_head,
    stride_K_msa,
    stride_K_seq,
    stride_K_dim,
    
    stride_V_batch,
    stride_V_head,
    stride_V_msa,
    stride_V_seq,
    stride_V_dim,
    
    stride_O_batch,
    stride_O_head,
    stride_O_msa,
    stride_O_seq,
    stride_O_dim,
    
    stride_pair_bias_batch,
    stride_pair_bias_head,
    stride_pair_bias_seq1,
    stride_pair_bias_seq2,
    
    stride_mask_batch,
    stride_mask_msa,
    stride_mask_seq,
    
    BATCH_SIZE,
    HEAD: tl.constexpr,
    N_SEQ: tl.constexpr, 
    SEQ_LEN: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    index_batch_head_msa = tl.program_id(1)
    index_batch_head = index_batch_head_msa // N_SEQ
    index_msa = index_batch_head_msa % N_SEQ
    index_batch = index_batch_head // HEAD
    index_head = index_batch_head % HEAD
    
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
        + index_msa.to(tl.int64) * stride_Q_msa
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )
    
    pair_bias_block_ptr = tl.make_block_ptr(
        base=pair_bias + (index_batch.to(tl.int64) * stride_pair_bias_batch + index_head.to(tl.int64) * stride_pair_bias_head),
        shape=(SEQ_LEN, SEQ_LEN),
        strides=(stride_pair_bias_seq1, stride_pair_bias_seq2),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KV),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        
    res_mask_block_ptr = res_mask + (index_batch.to(tl.int64) * stride_mask_batch + index_msa.to(tl.int64) * stride_mask_msa) + (offs_kv[None, :] * stride_mask_seq)

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)
    
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        res_mask_block_ptr,
        pair_bias_block_ptr,
        block_index_q,
        stride_mask_seq,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        4 - STAGE,
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head_msa * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head_msa = tl.program_id(1)
    offs_dim = tl.arange(0, DIM)
    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(
        O
        + index_batch_head_msa * SEQ_LEN * DIM 
        + offs_q[:, None] * DIM
        + offs_dim[None, :]
    )
    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO
        + index_batch_head_msa * SEQ_LEN * DIM 
        + offs_q[:, None] * DIM
        + offs_dim[None, :]
    ).to(tl.float32)
    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
    # Store the D block
    D_block_ptrs = D + index_batch_head_msa * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    res_mask,
    pair_bias,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    d_pair_bias,
    M,
    D,
    stride_batch,
    stride_head,
    stride_msa,
    stride_seq,
    stride_dim,
    stride_pair_bias_batch,
    stride_pair_bias_head,
    stride_pair_bias_seq1,
    stride_pair_bias_seq2,
    stride_mask_batch,
    stride_mask_msa,
    stride_mask_seq,
    stride_d_pair_bias_batch,
    stride_d_pair_bias_head,
    stride_d_pair_bias_seq1,
    stride_d_pair_bias_seq2,
    HEAD,
    N_SEQ,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head_msa = tl.program_id(2)
    index_batch_head = index_batch_head_msa // N_SEQ
    index_msa = index_batch_head_msa % N_SEQ
    index_batch = index_batch_head // HEAD
    index_head = index_batch_head % HEAD
    
    offset_batch_head_msa = (index_batch * stride_batch + index_head * stride_head + index_msa * stride_msa).to(tl.int64)
    offset_batch_head_msa_seq = (index_batch_head_msa * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head_msa
    K += offset_batch_head_msa
    V += offset_batch_head_msa
    dO += offset_batch_head_msa
    dQ += offset_batch_head_msa
    dK += offset_batch_head_msa
    dV += offset_batch_head_msa

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_msa_seq
    D += offset_batch_head_msa_seq

    # load scales
    offs_dim = tl.arange(0, DIM)

    index_block_kv = tl.program_id(0)
    offs_q = index_block_kv * BLOCK_Q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]
    
    pair_bias_block_ptr = tl.make_block_ptr(
        base=pair_bias + (index_batch.to(tl.int64) * stride_pair_bias_batch + index_head.to(tl.int64) * stride_pair_bias_head),
        shape=(SEQ_LEN, SEQ_LEN),
        strides=(stride_pair_bias_seq1, stride_pair_bias_seq2),
        offsets=(index_block_kv * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_KV),
        order=(1, 0),
    )
    offs_kv = tl.arange(0, BLOCK_KV)
    
    assert d_pair_bias.is_contiguous()
    d_pair_bias_block_ptr = d_pair_bias + (index_batch * stride_d_pair_bias_batch + index_head * stride_d_pair_bias_head).to(tl.int64) + (offs_q[:, None] * stride_d_pair_bias_seq1) + (offs_kv[None, :] * stride_d_pair_bias_seq2)

    res_mask_block_ptr = res_mask + (index_batch.to(tl.int64) * stride_mask_batch + index_msa.to(tl.int64) * stride_mask_msa) + (offs_kv[None, :] * stride_mask_seq)

    # We access the K and V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    Q_block = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)
    
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        pair_bias_block = tl.load(pair_bias_block_ptr)
        res_mask_block = tl.load(res_mask_block_ptr).broadcast_to((BLOCK_Q, BLOCK_KV))
        QK_block = tl.dot(Q_block, K_T_block) + pair_bias_block
        
        QK_block = tl.where(res_mask_block, QK_block, get_neg_max_value(QK_block.dtype))
        
        P_block = tl.math.exp(QK_block - M_block)

        # Compute dP and dS.
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        
        # Update d_pair_bias atomic add with float32 precision 
        tl.atomic_add(d_pair_bias_block_ptr, dS_block)
        
        dS_block = dS_block.to(K_T_block.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Increment pointers.
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq
        pair_bias_block_ptr = tl.advance(pair_bias_block_ptr, (0, BLOCK_KV))
        d_pair_bias_block_ptr += BLOCK_KV  * stride_d_pair_bias_seq2
        res_mask_block_ptr += BLOCK_KV * stride_mask_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    res_mask,
    pair_bias,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_msa,
    stride_seq,
    stride_dim,
    
    stride_pair_bias_batch,
    stride_pair_bias_head,
    stride_pair_bias_seq1,
    stride_pair_bias_seq2,
    
    stride_mask_batch,
    stride_mask_msa,
    stride_mask_seq,
    
    HEAD,
    N_SEQ, 
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head_msa = tl.program_id(2)
    index_batch_head = index_batch_head_msa // N_SEQ
    index_msa = index_batch_head_msa % N_SEQ
    index_batch = index_batch_head // HEAD
    index_head = index_batch_head % HEAD
    
    offset_batch_head_msa = (index_batch * stride_batch + index_head * stride_head + index_msa * stride_msa).to(tl.int64)
    offset_batch_head_msa_seq = (index_batch_head_msa * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head_msa
    K += offset_batch_head_msa
    V += offset_batch_head_msa
    dO += offset_batch_head_msa
    dQ += offset_batch_head_msa
    dK += offset_batch_head_msa
    dV += offset_batch_head_msa

    M += offset_batch_head_msa_seq
    D += offset_batch_head_msa_seq

    # load scales
    offs_dim = tl.arange(0, DIM)

    index_block_kv = tl.program_id(0)
    offs_kv = index_block_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)

    dK_block = tl.zeros([BLOCK_KV, DIM], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_KV, DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV, DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV, DIM)
    
    offs_q = tl.arange(0, BLOCK_Q)
    
    # pointer to tranposed pair bias
    pair_bias_T_block_ptr = pair_bias + (index_batch.to(tl.int64) * stride_pair_bias_batch + index_head.to(tl.int64) * stride_pair_bias_head) + offs_q[None, :] * stride_pair_bias_seq1 + offs_kv[:, None] * stride_pair_bias_seq2

    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector ans offs_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    res_mask_block_ptr = res_mask + (index_batch.to(tl.int64) * stride_mask_batch + index_msa.to(tl.int64) * stride_mask_msa) + offs_kv[None, :] * stride_mask_seq
    res_mask_T_block = tl.trans(tl.load(res_mask_block_ptr)).broadcast_to((BLOCK_KV, BLOCK_Q))

    K_block = K_block * tl.full((1,), softmax_scale, dtype=K_block.dtype)
    
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)
        pair_bias_T_block = tl.load(pair_bias_T_block_ptr)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = P^T
        QK_T_block = tl.dot(K_block, qT_block) + pair_bias_T_block
        
        # apply mask:
        QK_T_block = tl.where(res_mask_T_block, QK_T_block, get_neg_max_value(QK_T_block.dtype))
                
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        dO_block = tl.load(dO_ptrs)
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(K_block.dtype), dO_block)

        # Delta = rowsum(O * dO) where * is the element-wise product
        Di = tl.load(D + offs_q)

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(K_block.dtype)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq
        pair_bias_T_block_ptr += BLOCK_Q * stride_pair_bias_seq1

    # Write back dV.
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK.
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


class EvoformerAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, res_mask=None, pair_bias=None):
        # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
        # res_mask: [Batch, N_seq, 1, 1, N_res]
        # pair_bias: [Batch, 1, Head, N_res, N_res]
        mem_usage = SynchronizedWallClockTimer.memory_usage()
        print("Memory usage after init: ", mem_usage)
        
        DIM_Q, DIM_K, DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        assert DIM_Q == DIM_K and DIM_K == DIM_V
        
        BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM = Q.shape
        softmax_scale = DIM ** -0.5
        
        Q = Q.reshape((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM))
        K = K.reshape((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM))
        V = V.reshape((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM))
        res_mask = res_mask.reshape((BATCH_SIZE, 1, N_SEQ, SEQ_LEN, 1))
        pair_bias = pair_bias.reshape((BATCH_SIZE, HEAD, 1, SEQ_LEN, SEQ_LEN))  
        
        O = torch.empty_like(Q)
        stage = 1
        
        # Tuning for AMD target
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * HEAD * N_SEQ,
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        
        mem_usage = SynchronizedWallClockTimer.memory_usage()
        print("Memory usage after init O and M: ", mem_usage)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_msa=Q.stride(2),
            stride_Q_seq=Q.stride(3),
            stride_Q_dim=Q.stride(4),
            
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_msa=K.stride(2),
            stride_K_seq=K.stride(3),
            stride_K_dim=K.stride(4),
            
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_msa=V.stride(2),
            stride_V_seq=V.stride(3),
            stride_V_dim=V.stride(4),
            
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_msa=O.stride(2),
            stride_O_seq=O.stride(3),
            stride_O_dim=O.stride(4),
            
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(1),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(2),
            stride_mask_seq=res_mask.stride(3),
            
            BATCH_SIZE=BATCH_SIZE,
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            DIM=DIM,
            STAGE=stage,
            **extra_kern_args
        )
        
        mem_usage = SynchronizedWallClockTimer.memory_usage()
        print("Memory usage after fwd pass: ", mem_usage)

        ctx.save_for_backward(Q, K, V, res_mask, pair_bias, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.DIM = DIM_K
        
        # change back to original shape 
        O = O.reshape((BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM))
        
        return O
    

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, res_mask, pair_bias, O, M = ctx.saved_tensors
        
        BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM = dO.shape
        dO = dO.reshape((BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM)) 

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        # review: torch.float32 -- is it initialized this way # review: torch.zeros
        d_pair_bias = torch.empty((BATCH_SIZE, HEAD, SEQ_LEN, SEQ_LEN), device=pair_bias.device, dtype=torch.float32).zero_()
        
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        assert SEQ_LEN % 128 == 0, "Sequence length must be divided by 128"
        
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * HEAD * N_SEQ)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN)

        # Compute all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            DIM=DIM,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * HEAD * N_SEQ)

        stage = 1

        # Fix KV and iterate through all the Q blocks
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_msa=Q.stride(2),
            stride_seq=Q.stride(3),
            stride_dim=Q.stride(4),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(1),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(2),
            stride_mask_seq=res_mask.stride(3),
            
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            DIM=ctx.DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            d_pair_bias=d_pair_bias,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_msa=Q.stride(2),
            stride_seq=Q.stride(3),
            stride_dim=Q.stride(4),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(1),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(2),
            stride_mask_seq=res_mask.stride(3),
            stride_d_pair_bias_batch=d_pair_bias.stride(0),
            stride_d_pair_bias_head=d_pair_bias.stride(1),
            stride_d_pair_bias_seq1=d_pair_bias.stride(2),
            stride_d_pair_bias_seq2=d_pair_bias.stride(3),
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            DIM=ctx.DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        
        # change shape to original 
        dQ = dQ.reshape((BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM))
        dK = dK.reshape((BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM))
        dV = dV.reshape((BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM))
        d_pair_bias = d_pair_bias.reshape((BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN)).to(dO.dtype)

        return dQ, dK, dV, None, d_pair_bias

