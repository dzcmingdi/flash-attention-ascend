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
