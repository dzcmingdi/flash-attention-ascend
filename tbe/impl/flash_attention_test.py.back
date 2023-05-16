from tbe import tik
import tbe.common.platform as tbe_platform
import numpy
from impl.util.util_gemm import gemm_compute

DTYPE_SIZE = {
    'int8': 1,
    'float16': 2,
    'float32': 4,
}


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
            # with tik_instance.if_scope(rep_times == 1):
            dst_rep_stride = 0
            src_rep_stride = 0

        tik_instance.vec_trans_scatter(False, False, dst_list, src_list, rep_times, dst_rep_stride, src_rep_stride)
        tik_instance.data_move(k1nk0_tensor[index * k0 * n], k1nk0_ub, 0, 1, burst_len, 0, 0)


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


def flash_attention(q_type, k_type, v_type, o_type, kernel_name="FlashAttention"):
    tik_instance = tik.Tik(disable_debug=False)

    batch_size = q_type['shape'][0]
    assert q_type['shape'][0] == k_type['shape'][0] and k_type['shape'][0] == v_type['shape'][0]

    # q_shape = q_type['shape']
    # k_shape = k_type['shape']
    # v_shape = v_type['shape']
    q_shape = q_type['shape'][1:]
    k_shape = k_type['shape'][1:]
    v_shape = v_type['shape'][1:]
    o_shape = o_type['shape'][1:]
    assert q_shape[1] == k_shape[0]
    assert q_shape[1] % 16 == 0
    assert k_shape[0] % 16 == 0
    assert k_shape[1] % 16 == 0
    # assert o_shape[0] == q_shape[0]
    # assert o_shape[1] == v_shape[1]
    k0 = 16
    m0 = 16
    n0 = 16
    k = q_shape[1]
    k1 = (k + k0 - 1) // k0
    m = ((q_shape[0] + m0 - 1) // m0) * m0
    n1 = (k_shape[1] + n0 - 1) // 16
    n = n1 * n0

    d = k1 * k0

    q = tik_instance.Tensor(dtype='float16', shape=q_type['shape'], name="q", scope=tik.scope_gm)
    q_ = tik_instance.Tensor(dtype='float16', shape=[k1, m, k0], name="q_", scope=tik.scope_gm, is_workspace=True)

    k = tik_instance.Tensor(dtype='float16', shape=k_type['shape'], name='k', scope=tik.scope_gm)

    k_ = tik_instance.Tensor(dtype='float16', shape=[k1, n, k0], name='k_', scope=tik.scope_gm, is_workspace=True)
    v = tik_instance.Tensor(dtype='float16', shape=v_type['shape'], name='v', scope=tik.scope_gm)

    o = tik_instance.Tensor(dtype='float16', shape=[batch_size, n1, m, n0], name='sij', scope=tik.scope_gm)
    with tik_instance.for_range(0, 2, block_num=2) as c:
        with tik_instance.for_range(0, batch_size // 2, thread_num=2) as i:
            b = c * (batch_size // 2) + i
            o_ = o[b * (m * n): (b + 1) * (m * n)]
            with tik_instance.new_stmt_scope(disable_sync=True):
                transpose_left_matrix(tik_instance, q[b * m * d: (b + 1) * m * d], q_, 'float16', k1, m, k0)
                transpose_right_matrix(tik_instance, k[b * m * d: (b + 1) * m * d], k_, 'float16', k1, n, k0)
            # gemm_compute(q_, k_, o_, {})
            block_matmul(q_, k_, o_, [k1, m, k0], [k1, n, k0], [n1, m, n0], tik_instance)

    return tik_instance.BuildCCE(kernel_name="FlashAttention", inputs=[q, k, v], outputs=[o])


if __name__ == '__main__':
    tbe_platform.set_current_compile_soc_info("Ascend310")

    d = 192

    n = 128
    input_q = numpy.random.random((16, n, d)).astype(numpy.float16)

    input_k = numpy.random.random((16, d, n)).astype(numpy.float16)

    input_v = numpy.random.random((16, n, d)).astype(numpy.float16)

    feed_dict = {'q': input_q, 'k': input_k, 'v': input_v}
    #
    tik_instance = flash_attention({'shape': input_q.shape}, {'shape': input_k.shape}, {'shape': input_v.shape},
                                   {'shape': [input_q.shape[0], input_q.shape[1], input_v.shape[2]]})

    og, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
