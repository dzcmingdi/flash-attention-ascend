from tbe import tik
import tbe.common.platform as tbe_platform
import numpy

# from tbe.common.utils import para_check

ai_core_num = tbe_platform.get_soc_spec("CORE_NUM")
L1_BUFFER_SIZE = tbe_platform.get_soc_spec("L1_SIZE")

def block_matmul(q_shape, k_shape, q, k, out, tik_instance):
    M = q_shape[1]

    N = k_shape[1]

    K1 = q_shape[0]
    K0 = 16

    q_cbuf = tik_instance.Tensor(dtype='float16', shape=q_shape, name='q_cbuf', scope=tik.scope_cbuf)
    k_cbuf = tik_instance.Tensor(dtype='float16', shape=k_shape, name='k_cbuf', scope=tik.scope_cbuf)

    tik_instance.data_move(q_cbuf, q, 0, 1, (q_shape[0] * q_shape[1] * q_shape[2] * 2) // 32, 0, 0)
    tik_instance.data_move(k_cbuf, k, 0, 1, (k_shape[0] * k_shape[1] * k_shape[2] * 2) // 32, 0, 0)

    # input_q_cbuf = input_q_cbuf.reshape([2, 16, 16])
    # input_k_cbuf = input_k_cbuf.reshape([2, 16, 16])

    dst_ubuf = tik_instance.Tensor(dtype='float32', shape=[N // 16, M // 16 * 16, 16], name='dst_ubuf',
                                   scope=tik.scope_cbuf_out)
    tik_instance.matmul(dst_ubuf, q_cbuf, k_cbuf, M, K0 * K1, N)

    output = tik_instance.Tensor(dtype='float16', shape=dst_ubuf.shape, name='output', scope=tik.scope_gm)

    tik_instance.fixpipe(output, dst_ubuf, N // 16, (M * 64) // 32, 0, 0,
                         {"quantize_params": {"mode": "fp322fp16", "mode_param": None}})

    return output


def flash_attention(q_type, k_type, kernel_name="FlashAttention"):
    tik_instance = tik.Tik(disable_debug=False)

    q_shape = q_type['shape']
    k_shape = k_type['shape']
    q = tik_instance.Tensor(dtype='float16', shape=q_type['shape'], name="q", scope=tik.scope_gm)
    k = tik_instance.Tensor(dtype='float16', shape=k_type['shape'], name='k', scope=tik.scope_gm)

    tik_instance.BuildCCE(kernel_name="FlashAttention", inputs=[q, k], outputs=[])
    return tik_instance


if __name__ == '__main__':
    tbe_platform.set_current_compile_soc_info("Ascend310")
    print(L1_BUFFER_SIZE)


    # input_q = numpy.random.random(size=(16, 32)).astype(numpy.float16)
    # input_k = numpy.random.random(size=(32, 16)).astype(numpy.float16)
    #
    # input_q = input_q.reshape((16, 2, 16)).transpose((1, 0, 2))
    # input_k = input_k.reshape((2, 16, 16)).transpose((0, 2, 1))
    # feed_dict = {'q': input_q, 'k': input_k}
    #
    # tik_instance = flash_attention({'shape': input_q.shape}, {'shape': input_k.shape})
    #
    # 启动功能调试
    # o, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    # print(o)
    # 打印输出数据
