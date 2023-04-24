from tbe import tik
import tbe.common.platform as tbe_platform
import numpy

# from tbe.common.utils import para_check
DTYPE_SIZE = {
    'int8': 1,
    'float16': 2,
    'float32': 4,
}

ai_core_num = tbe_platform.get_soc_spec("CORE_NUM")
L1_BUFFER_SIZE = tbe_platform.get_soc_spec("L1_SIZE")
U_BUFFER_SIZE = tbe_platform.get_soc_spec("UB_SIZE")


# M*K -> K1 * M * K0
def transpose_left_matrix(tik_instance, mk_input_tensor, k1mk0_tensor, dtype, k1, m, k0):
    """change data format mk to k1mk0"""
    src_ub = tik_instance.Tensor(dtype, (k1, m, k0), name="src_ub", scope=tik.scope_ubuf)

    # data_move(m,k) --> (k1,m,k0)
    with tik_instance.for_range(0, k1) as i:
        tik_instance.data_move(src_ub[i * m * k0:], mk_input_tensor[i * k0:], 0, m, k0 * DTYPE_SIZE[dtype] // 32,
                               (k1 - 1) * k0 * DTYPE_SIZE[dtype] // 32, 0)
    # data_move out
    tik_instance.data_move(k1mk0_tensor, src_ub, 0, 1, k1 * m * k0 * DTYPE_SIZE[dtype] // 32, 0, 0)


# K*N -> K1 * N * K0
def transpose_right_matrix(tik_instance, kn_input_tensor, k1nk0_tensor, dtype, k1, n, k0):
    with tik_instance.for_range(0, k1) as index:
        k1nk0_ub = tik_instance.Tensor(dtype, (n, k0), tik.scope_ubuf, "k1nk0_ub")
        src_ub = tik_instance.Tensor(dtype, (k0, n), tik.scope_ubuf, "src_ub")
        burst_len = k0 * n * DTYPE_SIZE[dtype] // 32
        tik_instance.data_move(src_ub, kn_input_tensor[index * k0 * n], 0, 1, burst_len, 0, 0)
        dst_list = [k1nk0_ub[16 * i] for i in range(16)]
        src_list = [src_ub[n * i] for i in range(16)]
        rep_times = n // k0

        dst_rep_stride = k0
        src_rep_stride = 1

        if rep_times == 1:  # TODO: to be dynamic
            dst_rep_stride = 0
            src_rep_stride = 0

        tik_instance.vec_trans_scatter(False, False, dst_list, src_list, rep_times, dst_rep_stride, src_rep_stride)
        tik_instance.data_move(k1nk0_tensor[index * k0 * n], k1nk0_ub, 0, 1, burst_len, 0, 0)


# N1 * M * N0 -> M * N
def transpose_output_matrix(tik_instance, mn_output_tensor, n1mn0_tensor, dtype, n1, m, n0):
    src_ub = tik_instance.Tensor(dtype, (m, n1 * n0), name="src_ub", scope=tik.scope_ubuf)

    # data_move (n1,m,n0) --> (m,n)
    with tik_instance.for_range(0, n1) as i:
        tik_instance.data_move(src_ub[i * n0:], n1mn0_tensor[i * m * n0:], 0, m,
                               n0 * DTYPE_SIZE[dtype] // 32, 0, (n1 - 1) * n0 * DTYPE_SIZE[dtype] // 32)
    # data_move out
    tik_instance.data_move(mn_output_tensor, src_ub, 0, 1, m * n1 * n0 * DTYPE_SIZE[dtype] // 32, 0, 0)


def block_matmul(qt, kt, ot, tik_instance):
    q_shape = qt.shape
    k_shape = kt.shape
    o_shape = ot.shape

    q_cbuf = tik_instance.Tensor(dtype='float16', shape=qt.shape, name='q_cbuf', scope=tik.scope_cbuf)
    k_cbuf = tik_instance.Tensor(dtype='float16', shape=kt.shape, name='k_cbuf', scope=tik.scope_cbuf)

    tik_instance.data_move(q_cbuf, qt, 0, 1, (q_shape[0] * q_shape[1] * q_shape[2] * 2) // 32, 0, 0)
    tik_instance.data_move(k_cbuf, kt, 0, 1, (k_shape[0] * k_shape[1] * k_shape[2] * 2) // 32, 0, 0)

    # input_q_cbuf = input_q_cbuf.reshape([2, 16, 16])
    # input_k_cbuf = input_k_cbuf.reshape([2, 16, 16])

    dst_ubuf = tik_instance.Tensor(dtype='float32', shape=o_shape, name='dst_ubuf', scope=tik.scope_cbuf_out)
    tik_instance.matmul(dst_ubuf, q_cbuf, k_cbuf, q_shape[1], q_shape[0] * q_shape[2], k_shape[1])

    tik_instance.fixpipe(ot, dst_ubuf, ot.shape[0], (ot.shape[1] * ot.shape[2] * 4) // 32, 0, 0,
                         {"quantize_params": {"mode": "fp322fp16", "mode_param": None}})


def init_value(_tik_instance, input_gm_tensor, input_size, value):
    input_shape = input_gm_tensor.shape

    input_ub_tensor = _tik_instance.Tensor(dtype='float16', shape=input_shape, name='O_ub', scope=tik.scope_ubuf)

    # tik_instance.data_move(input_ub_tensor, input_gm_tensor, 0, 1, )

    src_scalar = _tik_instance.Scalar(init_value=value, dtype='float16')

    mask = 128

    strides = input_size // 16

    repeat_times = (input_size + 127) // mask

    assert strides % repeat_times == 0

    stride = strides // repeat_times

    _tik_instance.vec_dup(stride * 16, input_ub_tensor, src_scalar, repeat_times, stride)

    _tik_instance.data_move(input_gm_tensor, input_ub_tensor, 0, 1, input_size // 16, 0, 0)


def row_max(_matrix, output, _tik_instance):
    shape = _matrix.shape
    c1, r, c0 = shape[0], shape[1], shape[2]
    assert c0 == 16

    # _matrix = _tik_instance.Tensor(dtype='float16', shape=[c1, r, c0], name='_matrix', scope=tik.scope_ubuf)
    _matrix_T = _tik_instance.Tensor(dtype='float16', shape=[c0, r], name='_matrix_T', scope=tik.scope_ubuf)

    src_scalar = _tik_instance.Scalar(init_value=-65504., dtype='float16')
    mask = 128
    strides = r // 16
    repeat_times = (r + 127) // mask
    assert strides % repeat_times == 0
    stride = strides // repeat_times
    _tik_instance.vec_dup(stride * 16, output, src_scalar, repeat_times, stride)

    with _tik_instance.for_range(0, c1) as c1j:
        _tik_instance.vec_trans(_matrix_T, _matrix[c1j * r * c0: (c1j + 1) * r * c0], 1, 0, 0)

        _tik_instance.vec_max(stride * 16, output, _matrix_T, output, repeat_times, 0, stride, 0)


def row_sum(_matrix, output, _tik_instance):  # TODO exist overflow problem
    shape = _matrix.shape
    br, bc = shape[0], shape[1]

    r = br // 16
    c = bc // 16
    assert br <= 128
    assert bc <= 128
    _block_matrix_T = _tik_instance.Tensor(dtype='float16', shape=[16, bc], name='_block_matrix', scope=tik.scope_ubuf)
    _block_matrix_sum = _tik_instance.Tensor(dtype='float16', shape=[bc], name='_block_matrix_sum',
                                             scope=tik.scope_ubuf)
    _16x16_matrix = _tik_instance.Tensor(dtype='float16', shape=[16, 16], name='_16x16_matrix', scope=tik.scope_ubuf)
    src_scalar = _tik_instance.Scalar(init_value=0., dtype='float16')
    _tik_instance.vec_dup(br, output, src_scalar, 1, 0)
    _tik_instance.vec_dup(bc, _block_matrix_sum, src_scalar, 1, 0)

    with _tik_instance.for_range(0, r) as i:
        with _tik_instance.for_range(0, c) as j:
            _tik_instance.data_move(_16x16_matrix, _matrix[i * (16 * bc) + j * 16:], 0, 16, 1, (c - 1), 0)
            _tik_instance.vec_trans(_16x16_matrix, _16x16_matrix, 1, 1, 1)
            _tik_instance.data_move(_block_matrix_T[j * 16:], _16x16_matrix, 0, 16, 1, 0, (c - 1))
        with _tik_instance.for_range(0, 16) as z:
            _tik_instance.vec_add(bc, _block_matrix_sum, _block_matrix_T[z, :], _block_matrix_sum, 1, 0, 0, 0)

        _tik_instance.vec_add(16, output[i * 16: (i + 1) * 16], _block_matrix_sum,
                              output[i * 16: (i + 1) * 16], c, 0, 1, 0)


def flash_attention(q_type, k_type, v_type, kernel_name="FlashAttention"):
    tik_instance = tik.Tik(disable_debug=False)

    q_shape = q_type['shape']
    k_shape = k_type['shape']
    assert q_shape[1] == k_shape[0]
    assert q_shape[1] % 16 == 0
    assert q_shape[0] % 16 == 0
    assert k_shape[1] % 16 == 0
    k0 = 16
    m0 = 16
    n0 = 16
    k = q_shape[1]
    k1 = (k + k0 - 1) // k0
    m = ((q_shape[0] + m0 - 1) // m0) * m0
    n1 = (k_shape[1] + n0 - 1) // 16
    n = n1 * n0

    q = tik_instance.Tensor(dtype='float16', shape=q_type['shape'], name="q", scope=tik.scope_gm)
    q_ = tik_instance.Tensor(dtype='float16', shape=[k1, m, k0], name="q_", scope=tik.scope_gm, is_workspace=True)

    k = tik_instance.Tensor(dtype='float16', shape=k_type['shape'], name='k', scope=tik.scope_gm)
    k_ = tik_instance.Tensor(dtype='float16', shape=[k1, n, k0], name='k_', scope=tik.scope_gm, is_workspace=True)

    v = tik_instance.Tensor(dtype='float16', shape=v_type['shape'], name='v', scope=tik.scope_gm)
    v_ = tik_instance.Tensor(dtype='float16', shape=[k1, n, k0], name='v_', scope=tik.scope_gm, is_workspace=True)

    O = tik_instance.Tensor(dtype='float16', shape=[k1, m, k0], name='O', scope=tik.scope_gm)
    L = tik_instance.Tensor(dtype='float16', shape=[m], name='L', scope=tik.scope_gm)
    M = tik_instance.Tensor(dtype='float16', shape=[m], name='M', scope=tik.scope_gm)

    init_value(tik_instance, O, k1 * m * k0, value=0.)
    init_value(tik_instance, L, m, value=0.)
    init_value(tik_instance, M, m, -65504.)

    block_size_r = 16
    tr = m // block_size_r

    block_size_c = 16
    tc = n // block_size_c

    assert m % block_size_r == 0
    assert n % block_size_c == 0
    assert block_size_c // 16
    sij = tik_instance.Tensor(dtype='float16', shape=[block_size_c // 16, block_size_r, 16], name='sij',
                              scope=tik.scope_gm, is_workspace=True)
    with tik_instance.for_range(0, tc) as j:
        # kj = tik_instance.Tensor(dtype='float16', shape=[k1, block_size_c, k0], name='kj', scope=tik.scope_gm)
        # vj = tik_instance.Tensor(dtype='float16', shape=[k1, block_size_c, k0], name='vj', scope=tik.scope_gm)
        kj = k_[:, j * block_size_c: (j + 1) * block_size_c, :]
        vj = v_[:, j * block_size_c: (j + 1) * block_size_c, :]

        with tik_instance.for_range(0, tr) as i:
            qi = q_[:, i * block_size_r: (i + 1) * block_size_r, :]

            block_matmul(qi, kj, sij, tik_instance)

            sij_ub = tik_instance.Tensor(dtype='float16', shape=[block_size_c // 16, block_size_r, 16], name='sij',
                                         scope=tik.scope_ubuf)

            tik_instance.data_move(sij_ub, sij, 0, 1, (block_size_r * block_size_c) // 16, 0, 0)

            Mij = tik_instance.Tensor(dtype='float16', shape=[sij_ub.shape[1]], name='mij',
                                      scope=tik.scope_ubuf)
            row_max(sij_ub, Mij, tik_instance)

            pij = tik_instance.Tensor(dtype='float16', shape=[block_size_r, block_size_c], name='mij',
                                      scope=tik.scope_ubuf)
            with tik_instance.for_range(0, block_size_r) as br:
                row_max_oi = tik_instance.Scalar(dtype='float16', name='row_max_oi', init_value=Mij[br])
                tik_instance.vec_dup(block_size_c, pij[br, :], row_max_oi, 1, block_size_c // 16)

            p_mask = 128
            assert (block_size_r * block_size_c) % 128 == 0
            p_repeat_times = (block_size_r * block_size_c) // p_mask
            p_stride = p_mask // 16
            tik_instance.vec_sub(p_mask, pij, sij_ub, pij, p_repeat_times, p_stride, p_stride, p_stride)
            tik_instance.vec_exp(p_mask, pij, pij, p_repeat_times, p_stride, p_stride)

            Lij = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li', scope=tik.scope_ubuf)
            row_sum(pij, Lij, tik_instance)

            Mi = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Mi', scope=tik.scope_ubuf)
            Li = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li', scope=tik.scope_ubuf)

            Mi_new = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Mi_new', scope=tik.scope_ubuf)
            Li_new = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li_new', scope=tik.scope_ubuf)
            Li_sub0 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li_sub0', scope=tik.scope_ubuf)
            Li_sub1 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li_sub1', scope=tik.scope_ubuf)
            Li_exp0 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li_exp0', scope=tik.scope_ubuf)
            Li_exp1 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='Li_exp1', scope=tik.scope_ubuf)

            tik_instance.data_move(Mi, M, 0, 1, block_size_r // 16, 0, 0)
            tik_instance.data_move(Li, L, 0, 1, block_size_r // 16, 0, 0)

            tik_instance.vec_max(block_size_r, Mi_new, Mij, Mi, 1, 0, 0, 0)

            # TODO: can merge Mi and Mij into one tensor to accelerate.
            tik_instance.vec_sub(block_size_r, Li_sub0, Mi, Mi_new, 1, 0, 0, 0)
            tik_instance.vec_sub(block_size_r, Li_sub1, Mij, Mi_new, 1, 0, 0, 0)
            tik_instance.vec_exp(block_size_r, Li_sub0, Li_sub0, 1, 0, 0)
            tik_instance.vec_exp(block_size_r, Li_sub1, Li_sub1, 1, 0, 0)
            tik_instance.data_move(Li_exp0, Li_sub0, 0, 1, block_size_r // 16, 0, 0)
            tik_instance.data_move(Li_exp1, Li_sub1, 0, 1, block_size_r // 16, 0, 0)

            tik_instance.vec_mul(block_size_r, Li_sub0, Li, Li_sub0, 1, 0, 0, 0)
            tik_instance.vec_mul(block_size_r, Li_sub1, Lij, Li_sub1, 1, 0, 0, 0)
            tik_instance.vec_add(block_size_r, Li_new, Li_sub0, Li_sub1, 1, 0, 0, 0)

            r_diag_Li_new = tik_instance.Tensor(dtype='float16', shape=[block_size_r, block_size_r],
                                                name='r_diag_Li_new', scope=tik.scope_ubuf)

            with tik_instance.for_range(0, block_size_r) as z:
                r_diag_Li_new[z, z] = 1.

    transpose_left_matrix(tik_instance, q, q_, 'float16', k1, m, k0)
    transpose_right_matrix(tik_instance, k, k_, 'float16', k1, n, k0)

    # o_ = tik_instance.Tensor(dtype='float16', shape=[n1, m, n0], name='o_', scope=tik.scope_gm, is_workspace=True)
    # o = tik_instance.Tensor(dtype='float16', shape=[m, n], name='o', scope=tik.scope_gm)

    # block_matmul(q_, k_, o_, tik_instance)

    # transpose_output_matrix(tik_instance, o, o_, 'float16', n1, m, n0)

    tik_instance.BuildCCE(kernel_name="FlashAttention", inputs=[q, k], outputs=[O, L, M])
    return tik_instance


if __name__ == '__main__':
    tbe_platform.set_current_compile_soc_info("Ascend310")

    d = 192

    n = 64
    input_q = numpy.random.random(size=(n, d)).astype(numpy.float16)

    input_k = numpy.random.random(size=(d, n)).astype(numpy.float16)

    input_v = numpy.random.random(size=(n, d)).astype(numpy.float16)
    #
    # input_q = input_q.reshape((16, 2, 16)).transpose((1, 0, 2))
    # input_k = input_k.reshape((2, 16, 16)).transpose((0, 2, 1))

    # print(numpy.matmul(input_q, input_k))
    feed_dict = {'q': input_q, 'k': input_k}
    #
    tik_instance = flash_attention({'shape': input_q.shape}, {'shape': input_k.shape}, {'shape': input_v.shape})
    # #
    o, l, m = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    # print(o)
    print(l)

    print(m)
