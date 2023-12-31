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
    with tik_instance.for_range(0, k1, thread_num=2) as i:
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
            # with tik_instance.if_scope(rep_times == 1):
            dst_rep_stride = 0
            src_rep_stride = 0

        tik_instance.vec_trans_scatter(False, False, dst_list, src_list, rep_times, dst_rep_stride, src_rep_stride)
        tik_instance.data_move(k1nk0_tensor[index * k0 * n], k1nk0_ub, 0, 1, burst_len, 0, 0)


# k1 * n * k0 -> k * n
def r_transpose_right_matrix(tik_instance, kn_output_tensor, k1nk0_tensor, dtype, k1, n, k0):
    with tik_instance.for_range(0, k1) as index:
        k1k0n_ub = tik_instance.Tensor(dtype, (k0, n), tik.scope_ubuf, 'k1nk0_ub')
        src_ub = tik_instance.Tensor(dtype, (n, k0), tik.scope_ubuf, "src_ub")
        tik_instance.data_move(src_ub, k1nk0_tensor[index * n * k0], 0, 1, n * k0 * DTYPE_SIZE[dtype] // 32, 0, 0)
        dst_list = [k1k0n_ub[n * i] for i in range(16)]
        src_list = [src_ub[16 * i] for i in range(16)]
        rep_times = n // k0

        dst_rep_stride = 1
        src_rep_stride = k0

        if rep_times == 1:
            dst_rep_stride = 0
            src_rep_stride = 0
        tik_instance.vec_trans_scatter(False, False, dst_list, src_list, rep_times, dst_rep_stride, src_rep_stride)
        tik_instance.data_move(kn_output_tensor[index * n * k0], k1k0n_ub, 0, 1, n * k0 * DTYPE_SIZE[dtype] // 32, 0, 0)


# N1 * M * N0 -> M * N
def transpose_output_matrix(tik_instance, mn_output_tensor, n1mn0_tensor, dtype, n1, m, n0):
    src_ub = tik_instance.Tensor(dtype, (m, n1 * n0), name="src_ub", scope=tik.scope_ubuf)

    # data_move (n1,m,n0) --> (m,n)
    with tik_instance.for_range(0, n1) as i:
        tik_instance.data_move(src_ub[i * n0:], n1mn0_tensor[i * m * n0:], 0, m,
                               n0 * DTYPE_SIZE[dtype] // 32, 0, (n1 - 1) * n0 * DTYPE_SIZE[dtype] // 32)
    # data_move out
    tik_instance.data_move(mn_output_tensor, src_ub, 0, 1, m * n1 * n0 * DTYPE_SIZE[dtype] // 32, 0, 0)


def block_matmul(qt, kt, ot, q_shape, k_shape, o_shape, tik_instance):
    # q_shape = qt.shape
    # k_shape = kt.shape
    # o_shape = ot.shape

    q_cbuf = tik_instance.Tensor(dtype='float16', shape=qt.shape, name='q_cbuf', scope=tik.scope_cbuf)
    k_cbuf = tik_instance.Tensor(dtype='float16', shape=kt.shape, name='k_cbuf', scope=tik.scope_cbuf)

    tik_instance.data_move(q_cbuf, qt, 0, 1, (q_shape[0] * q_shape[1] * q_shape[2] * 2) // 32, 0, 0)
    tik_instance.data_move(k_cbuf, kt, 0, 1, (k_shape[0] * k_shape[1] * k_shape[2] * 2) // 32, 0, 0)

    # input_q_cbuf = input_q_cbuf.reshape([2, 16, 16])
    # input_k_cbuf = input_k_cbuf.reshape([2, 16, 16])

    dst_cbuf = tik_instance.Tensor(dtype='float32', shape=o_shape, name='dst_cbuf', scope=tik.scope_cbuf_out)
    tik_instance.matmul(dst_cbuf, q_cbuf, k_cbuf, q_shape[1], q_shape[0] * q_shape[2], k_shape[1])
    # tik_instance.tikdb.debug_print("dst_cbuf")

    tik_instance.fixpipe(ot, dst_cbuf, o_shape[0], (o_shape[1] * o_shape[2] * 4) // 32, 0, 0,
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
    c1, br, c0 = shape[0], shape[1], shape[2]
    assert c0 == 16
    assert br <= 128
    _block_matrix_max = _tik_instance.Tensor(dtype='float16', shape=[br, c0], name='_block_matrix_max',
                                             scope=tik.scope_ubuf)

    _16x16_matrix = _tik_instance.Tensor(dtype='float16', shape=[16, 16], name='_16x16_matrix', scope=tik.scope_ubuf)
    _block_matrix_T = _tik_instance.Tensor(dtype='float16', shape=[c0, br], name='_block_matrix_T',
                                           scope=tik.scope_ubuf)

    _tik_instance.vec_dup(128, _block_matrix_max, -65504., (br * c0) // 128, 8)
    _tik_instance.vec_dup(br, output, -65504., 1, 0)

    with _tik_instance.for_range(0, br // 8, thread_num=2) as i:
        # with _tik_instance.for_range(0, k1) as z:
        _tik_instance.vec_max(128, _block_matrix_max[i * 8:(i + 1) * 8, :], _matrix[i * 8 * c0],
                              _block_matrix_max[i * 8:(i + 1) * 8, :], c1, 0, br * c0 // 16, 0)

    dst_list = [_block_matrix_T[i * br] for i in range(c0)]
    src_list = [_block_matrix_max[i * c0] for i in range(c0)]

    repeat_times = br // c0

    if repeat_times == 1:
        _tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_times, 0, 0)
    else:
        _tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_times, 1, c0)
    _tik_instance.vec_max(br, output, _block_matrix_T, output, c0, 0, br // 16, 0)


def row_sum(_matrix, output, _tik_instance):  # TODO exist overflow problem
    shape = _matrix.shape
    k1, br, k0 = shape[0], shape[1], shape[2]

    assert br <= 128
    _block_matrix_sum = _tik_instance.Tensor(dtype='float16', shape=[br, k0], name='_block_matrix_sum',
                                             scope=tik.scope_ubuf)

    _16x16_matrix = _tik_instance.Tensor(dtype='float16', shape=[16, 16], name='_16x16_matrix', scope=tik.scope_ubuf)
    _block_matrix_T = _tik_instance.Tensor(dtype='float16', shape=[k0, br], name='_block_matrix_T',
                                           scope=tik.scope_ubuf)

    _tik_instance.vec_dup(128, _block_matrix_sum, 0., (br * k0) // 128, 8)
    _tik_instance.vec_dup(br, output, 0., 1, 0)

    with _tik_instance.for_range(0, br // 8, thread_num=2) as i:
        # with _tik_instance.for_range(0, k1) as z:
        _tik_instance.vec_add(128, _block_matrix_sum[i * 8:(i + 1) * 8, :], _matrix[i * 8 * k0],
                              _block_matrix_sum[i * 8:(i + 1) * 8, :], k1, 0, br * k0 // 16, 0)

    dst_list = [_block_matrix_T[i * br] for i in range(k0)]
    src_list = [_block_matrix_sum[i * k0] for i in range(k0)]

    repeat_times = br // k0

    if repeat_times == 1:
        _tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_times, 0, 0)
    else:
        _tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_times, 1, k0)
    _tik_instance.vec_add(br, output, _block_matrix_T, output, k0, 0, br // 16, 0)


def flash_attention(q_type, k_type, v_type, o_type, kernel_name="FlashAttention"):
    tik_instance = tik.Tik(disable_debug=False)

    batch_size = q_type['shape'][0]
    assert q_type['shape'][0] == k_type['shape'][0] and k_type['shape'][0] == v_type['shape'][0]

    q_shape = q_type['shape'][1:]
    k_shape = k_type['shape'][1:]
    v_shape = v_type['shape'][1:]
    o_shape = o_type['shape'][1:]
    assert q_shape[1] == k_shape[0]
    assert q_shape[1] % 16 == 0
    assert k_shape[0] % 16 == 0
    assert k_shape[1] % 16 == 0
    assert o_shape[0] == q_shape[0]
    assert o_shape[1] == v_shape[1]
    k0 = 16
    m0 = 16
    n0 = 16
    k = q_shape[1]
    k1 = (k + k0 - 1) // k0
    m = ((q_shape[0] + m0 - 1) // m0) * m0
    n1 = (k_shape[1] + n0 - 1) // 16
    n = n1 * n0

    d = k1 * k0

    block_size_r = 64
    tr = m // block_size_r

    block_size_c = 64
    tc = n // block_size_c

    assert m % block_size_r == 0
    assert n % block_size_c == 0
    assert block_size_c % 16 == 0

    q = tik_instance.Tensor(dtype='float16', shape=q_type['shape'], name="q", scope=tik.scope_gm)

    k = tik_instance.Tensor(dtype='float16', shape=k_type['shape'], name='k', scope=tik.scope_gm)

    v = tik_instance.Tensor(dtype='float16', shape=v_type['shape'], name='v', scope=tik.scope_gm)

    o = tik_instance.Tensor(dtype='float16', shape=[batch_size, m, d], name='o', scope=tik.scope_gm)

    batch_l = tik_instance.Tensor(dtype='float16', shape=[m], name='L', scope=tik.scope_gm, is_workspace=True)
    batch_m = tik_instance.Tensor(dtype='float16', shape=[m], name='M', scope=tik.scope_gm, is_workspace=True)

    cores_block_q = tik_instance.Tensor(dtype='float16', shape=[ai_core_num, k1, block_size_r, k0], name='block_q',
                                        scope=tik.scope_gm,
                                        is_workspace=True)
    block_k = tik_instance.Tensor(dtype='float16', shape=[k1, block_size_c, k0], name='block_k', scope=tik.scope_gm,
                                  is_workspace=True)
    block_k_origin = tik_instance.Tensor(dtype='float16', shape=[block_size_c, d], name='block_k_origin',
                                         scope=tik.scope_gm,

                                         is_workspace=True)
    block_v = tik_instance.Tensor(dtype='float16', shape=[block_size_c // k0, d, k0], name='block_v',
                                  scope=tik.scope_gm, is_workspace=True)

    cores_sij = tik_instance.Tensor(dtype='float16', shape=[ai_core_num, block_size_c // k0, block_size_r, k0],
                                    name='sij',
                                    scope=tik.scope_gm, is_workspace=True)

    cores_oi_mul0 = tik_instance.Tensor(dtype='float16', shape=[ai_core_num, k1, block_size_r, k0],
                                        name='cores_oi_mul0',
                                        scope=tik.scope_gm, is_workspace=True)

    cores_oi_mul1 = tik_instance.Tensor(dtype='float16', shape=[ai_core_num, k1, block_size_r, k0],
                                        name='cores_oi_mul1',
                                        scope=tik.scope_gm, is_workspace=True)

    cores_pij_gm = tik_instance.Tensor(dtype='float16', shape=[ai_core_num, block_size_c // k0, block_size_r, k0],
                                       name='pij_ub',
                                       scope=tik.scope_gm, is_workspace=True)

    with tik_instance.for_range(0, batch_size) as b:
        batch_o = o[b * m * d: (b + 1) * m * d]

        with tik_instance.for_range(0, tr, thread_num=2) as i:
            init_value(tik_instance, batch_o[i * block_size_r * d: (i + 1) * (block_size_r * d)], block_size_r * d,
                       value=0.)

        with tik_instance.new_stmt_scope(disable_sync=True):
            init_value(tik_instance, batch_l, m, value=0.)
            init_value(tik_instance, batch_m, m, -65504.)

        with tik_instance.for_range(0, tc) as j:
            with tik_instance.new_stmt_scope(disable_sync=False):
                block_k_ub = tik_instance.Tensor(dtype='float16', shape=block_k.shape, name='block_k_ub',
                                                 scope=tik.scope_ubuf)
                index_k = b * d * n + j * block_size_c

                tik_instance.data_move(block_k_ub, k[index_k], 0, d, block_size_c // 16, (n - block_size_c) // 16, 0)
                tik_instance.data_move(block_k_origin, block_k_ub, 0, 1, block_size_c * d // 16, 0, 0)

            with tik_instance.new_stmt_scope(disable_sync=True):
                index_v = b * d * n + j * block_size_c * d
                transpose_right_matrix(tik_instance, block_k_origin, block_k, 'float16', k1, block_size_c, k0)
                transpose_right_matrix(tik_instance, v[index_v: index_v + block_size_c * d], block_v, 'float16',
                                       block_size_c // k0, d, k0)

            with tik_instance.for_range(0, ai_core_num, block_num=ai_core_num) as core_i:
                block_q = cores_block_q[core_i * (block_size_r * d): (core_i + 1) * (block_size_r * d)]
                pij_gm = cores_pij_gm[
                         core_i * (block_size_r * block_size_c): (core_i + 1) * (block_size_r * block_size_c)]
                oi_mul1 = cores_oi_mul1[core_i * (block_size_r * d): (core_i + 1) * (block_size_r * d)]
                oi_mul0 = cores_oi_mul0[core_i * (block_size_r * d): (core_i + 1) * (block_size_r * d)]

                sij = cores_sij[core_i * (block_size_r * block_size_c): (core_i + 1) * (block_size_r * block_size_c)]
                with tik_instance.for_range(0, tr // ai_core_num) as z:
                    i = core_i * (tr // ai_core_num) + z
                    with tik_instance.new_stmt_scope(disable_sync=False):
                        index = b * m * d + i * block_size_r * d
                        transpose_left_matrix(tik_instance, q[index: index + block_size_r * d], block_q, 'float16', k1,
                                              block_size_r, k0)

                        block_matmul(block_q, block_k, sij, [k1, block_size_r, k0], block_k.shape,
                                     [block_size_c // k0, block_size_r, k0],
                                     tik_instance)

                    mij = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='mij',
                                              scope=tik.scope_ubuf)
                    #
                    lij = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='lij', scope=tik.scope_ubuf)

                    with tik_instance.new_stmt_scope(disable_sync=False):
                        sij_ub = tik_instance.Tensor(dtype='float16', shape=[block_size_c // k0, block_size_r, k0],
                                                     name='sij',
                                                     scope=tik.scope_ubuf)

                        tik_instance.data_move(sij_ub, sij, 0, 1, (block_size_r * block_size_c) // 16, 0, 0)
                        row_max(sij_ub, mij, tik_instance)

                        pij = tik_instance.Tensor(dtype='float16', shape=[block_size_c // k0, block_size_r, k0],
                                                  name='pij',
                                                  scope=tik.scope_ubuf)
                        with tik_instance.for_range(0, block_size_r) as br:
                            row_max_oi = tik_instance.Scalar(dtype='float16', name='row_max_oi', init_value=mij[br])
                            tik_instance.vec_dup(k0, pij[br * k0], row_max_oi, pij.shape[0], (block_size_r * k0) // 16)

                        p_mask = 128
                        assert (block_size_r * block_size_c) % 128 == 0
                        p_repeat_times = (block_size_r * block_size_c) // p_mask
                        p_stride = p_mask // 16
                        tik_instance.vec_sub(p_mask, pij, sij_ub, pij, p_repeat_times, p_stride, p_stride, p_stride)
                        tik_instance.vec_exp(p_mask, pij, pij, p_repeat_times, p_stride, p_stride)
                        row_sum(pij, lij, tik_instance)
                        tik_instance.data_move(pij_gm, pij, 0, 1, (block_size_c * block_size_r) // 16, 0, 0)

                    oi = tik_instance.Tensor(dtype='float16', shape=[block_size_r, d], name='oi',
                                             scope=tik.scope_ubuf)
                    tik_instance.data_move(oi, batch_o[i * block_size_r * d], 0, 1, block_size_r * d // 16, 0, 0)

                    oi_exp0 = tik_instance.Tensor(dtype='float16', shape=[k1, block_size_r, k0], name='oi_exp0',
                                                  scope=tik.scope_ubuf)
                    oi_exp1 = tik_instance.Tensor(dtype='float16', shape=[k1, block_size_r, k0], name='oi_exp1',
                                                  scope=tik.scope_ubuf)

                    r_diag_li_new = tik_instance.Tensor(dtype='float16', shape=[block_size_r, d],
                                                        name='r_diag_li_new', scope=tik.scope_ubuf)

                    r_diag_li = tik_instance.Tensor(dtype='float16', shape=[block_size_r, d],
                                                    name='r_diag_li',
                                                    scope=tik.scope_ubuf)

                    mi_new = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='mi_new',
                                                 scope=tik.scope_ubuf)
                    li_new = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li_new',
                                                 scope=tik.scope_ubuf)

                    with tik_instance.new_stmt_scope(disable_sync=False):
                        mi = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='mi', scope=tik.scope_ubuf)
                        li = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li', scope=tik.scope_ubuf)

                        li_new_rec = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li_new_rec',
                                                         scope=tik.scope_ubuf)
                        li_sub0 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li_sub0',
                                                      scope=tik.scope_ubuf)
                        li_sub1 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li_sub1',
                                                      scope=tik.scope_ubuf)
                        li_exp0 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li_exp0',
                                                      scope=tik.scope_ubuf)
                        li_exp1 = tik_instance.Tensor(dtype='float16', shape=[block_size_r], name='li_exp1',
                                                      scope=tik.scope_ubuf)

                        tik_instance.data_move(mi, batch_m[i * block_size_r], 0, 1, block_size_r // 16, 0, 0)
                        tik_instance.data_move(li, batch_l[i * block_size_r], 0, 1, block_size_r // 16, 0, 0)

                        tik_instance.vec_max(block_size_r, mi_new, mij, mi, 1, 0, 0, 0)
                        #
                        # # TODO: can merge Mi and Mij into one tensor to accelerate.
                        tik_instance.vec_sub(block_size_r, li_sub0, mi, mi_new, 1, 0, 0, 0)
                        tik_instance.vec_sub(block_size_r, li_sub1, mij, mi_new, 1, 0, 0, 0)
                        tik_instance.vec_exp(block_size_r, li_sub0, li_sub0, 1, 0, 0)
                        tik_instance.vec_exp(block_size_r, li_sub1, li_sub1, 1, 0, 0)
                        tik_instance.data_move(li_exp0, li_sub0, 0, 1, block_size_r // 16, 0, 0)
                        tik_instance.data_move(li_exp1, li_sub1, 0, 1, block_size_r // 16, 0, 0)

                        tik_instance.vec_mul(block_size_r, li_sub0, li, li_sub0, 1, 0, 0, 0)
                        tik_instance.vec_mul(block_size_r, li_sub1, lij, li_sub1, 1, 0, 0, 0)
                        tik_instance.vec_add(block_size_r, li_new, li_sub0, li_sub1, 1, 0, 0, 0)
                        tik_instance.vec_rec(block_size_r, li_new_rec, li_new, 1, 0, 0)

                        with tik_instance.for_range(0, block_size_r, thread_num=2) as r:
                            scalar0 = tik_instance.Scalar(dtype='float16', name='li_r', init_value=li[r])
                            scalar1 = tik_instance.Scalar(dtype='float16', name='li_new_r', init_value=li_new_rec[r])
                            scalar2 = tik_instance.Scalar(dtype='float16', name='oi_exp0_r', init_value=li_exp0[r])
                            scalar3 = tik_instance.Scalar(dtype='float16', name='oi_exp1_r', init_value=li_exp1[r])

                            if d // 128 != 0:
                                tik_instance.vec_dup(128, r_diag_li[r * d], scalar0, d // 128, 8)
                                tik_instance.vec_dup(128, r_diag_li_new[r * d], scalar1, d // 128, 8)

                                tik_instance.vec_dup(128, oi_exp0[r * d], scalar2, d // 128, 8)
                                tik_instance.vec_dup(128, oi_exp1[r * d], scalar3, d // 128, 8)

                            if d % 128 != 0:
                                start_index = r * d + (d // 128) * 128
                                tik_instance.vec_dup(d % 128, r_diag_li[start_index], scalar0, 1, 0)
                                tik_instance.vec_dup(d % 128, r_diag_li_new[start_index], scalar1, 1, 0)
                                tik_instance.vec_dup(d % 128, oi_exp0[start_index], scalar2, 1, 0)
                                tik_instance.vec_dup(d % 128, oi_exp1[start_index], scalar3, 1, 0)

                    tik_instance.vec_mul(128, r_diag_li, oi, r_diag_li, block_size_r * d // 128, 8, 8, 8)
                    block_matmul(pij_gm, block_v, oi_mul0, [block_size_c // k0, block_size_r, k0], block_v.shape,
                                 [k1, block_size_r, k0], tik_instance)

                    transpose_output_matrix(tik_instance, oi_mul1, oi_mul0, 'float16', k1, block_size_r, k0)
                    #
                    with tik_instance.new_stmt_scope(disable_sync=False):
                        oi_mul1_ub = tik_instance.Tensor(dtype='float16', shape=[k1, block_size_r, k0],
                                                         name='oi_mul1_ub',
                                                         scope=tik.scope_ubuf)
                        tik_instance.data_move(oi_mul1_ub, oi_mul1, 0, 1, (block_size_r * d) // 16, 0, 0)
                        tik_instance.vec_mul(128, r_diag_li, oi_exp0, r_diag_li, (block_size_r * d) // 128, 8, 8, 8)
                        tik_instance.vec_mul(128, oi_mul1_ub, oi_exp1, oi_mul1_ub, (block_size_r * d) // 128, 8, 8, 8)

                        tik_instance.vec_add(128, r_diag_li, oi_mul1_ub, r_diag_li, (block_size_r * d) // 128, 8,
                                             8,
                                             8)

                        tik_instance.vec_mul(128, oi, r_diag_li_new, r_diag_li, (block_size_r * d) // 128, 8, 8,
                                             8)

                    with tik_instance.new_stmt_scope(disable_sync=True):
                        tik_instance.data_move(batch_o[i * block_size_r * d], oi, 0, 1, block_size_r * d // 16, 0, 0)
                        tik_instance.data_move(batch_l[i * block_size_r], li_new, 0, 1, block_size_r // 16, 0, 0)
                        tik_instance.data_move(batch_m[i * block_size_r], mi_new, 0, 1, block_size_r // 16, 0, 0)
    tik_instance.BuildCCE(kernel_name="FlashAttention", inputs=[q, k, v], outputs=[o])

    return tik_instance


def get_numpy_result(q, k, v):
    results = []
    for bq, bk, bv in zip(q, k, v):
        s = numpy.matmul(bq, bk)
        m = numpy.max(s, axis=1, keepdims=True)
        p = numpy.exp(s - m)
        l = numpy.sum(p, axis=1, keepdims=True)
        scores = p / l
        results.append(numpy.matmul(scores, bv))

    return numpy.stack(results)
    # s = numpy.matmul(q, k)
    # m = numpy.max(s, axis=1, keepdims=True)
    # p = numpy.exp(s - m)
    # l = numpy.sum(p, axis=1, keepdims=True)
    # scores = p / l
    #
    # return numpy.matmul(scores, v)


if __name__ == '__main__':
    tbe_platform.set_current_compile_soc_info("Ascend310")

    # m = n = 16
    d = 96

    n = 64
    input_q = numpy.random.random((1, n, d)).astype(numpy.float16)

    input_k = numpy.random.random((1, d, n)).astype(numpy.float16)

    input_v = numpy.random.random((1, n, d)).astype(numpy.float16)

    input_q /= numpy.sqrt(numpy.linalg.norm(input_q, ord=1, axis=2, keepdims=True))
    input_k /= numpy.sqrt(numpy.linalg.norm(input_k, ord=1, axis=1, keepdims=True))
    input_v /= numpy.sqrt(numpy.linalg.norm(input_v, ord=1, axis=2, keepdims=True))

    feed_dict = {'q': input_q, 'k': input_k, 'v': input_v}
    #
    tik_instance = flash_attention({'shape': input_q.shape}, {'shape': input_k.shape}, {'shape': input_v.shape},
                                   {'shape': [input_q.shape[0], input_q.shape[1], input_v.shape[2]]})
    # #

    # [m // k0, d, k0]

    og, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    ot = get_numpy_result(input_q, input_k, input_v)
    print((og - ot))
