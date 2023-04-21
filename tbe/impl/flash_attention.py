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

    tik_instance.vec_dump(None, )

    transpose_left_matrix(tik_instance, q, q_, 'float16', k1, m, k0)
    transpose_right_matrix(tik_instance, k, k_, 'float16', k1, n, k0)

    # o_ = tik_instance.Tensor(dtype='float16', shape=[n1, m, n0], name='o_', scope=tik.scope_gm, is_workspace=True)
    # o = tik_instance.Tensor(dtype='float16', shape=[m, n], name='o', scope=tik.scope_gm)

    # block_matmul(q_, k_, o_, tik_instance)

    # transpose_output_matrix(tik_instance, o, o_, 'float16', n1, m, n0)

    tik_instance.BuildCCE(kernel_name="FlashAttention", inputs=[q, k], outputs=[o])
    return tik_instance


def init_value(tik_instance, input_gm_tensor, init_value):
    input_shape = input_gm_tensor.shape
    input_ub_tensor = tik_instance.Tensor(dtype='float16', shape=input_shape, name='O_ub', scope=tik.scope_gm)

    tik_instance.data_move(input_ub_tensor, input_gm_tensor, 0, 1, )
    tik_instance.vec_dup(None, tik_instance)


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

    print(numpy.matmul(input_q, input_k))
    feed_dict = {'q': input_q, 'k': input_k}
    #
    tik_instance = flash_attention({'shape': input_q.shape}, {'shape': input_k.shape}, {'shape': input_v.shape})
    # #
    o, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    print(o)
