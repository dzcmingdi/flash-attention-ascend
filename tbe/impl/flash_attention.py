from tbe import tik
import tbe.common.platform as tbe_platform
import numpy

# from tbe.common.utils import para_check

ai_core_num = tbe_platform.get_soc_spec("CORE_NUM")


def flash_attention(input_q, input_k, kernel_name="FlashAttention"):
    tbe_platform.set_current_compile_soc_info("Ascend310")
    tik_instance = tik.Tik(disable_debug=False)
    input_q = tik_instance.Tensor(dtype='float32', shape=[24], name="input_q", scope=tik.scope_gm)
    output = tik_instance.Tensor(dtype='float32', shape=[24], name='output', scope=tik.scope_gm)
    with tik_instance.for_range(0, 2, block_num=ai_core_num) as i:
        shape = 16 - (i * 16 + 16) // 24 * 8
        input_q_ubuf = tik_instance.Tensor(dtype='float32', shape=[shape], name="input_q_ubuf",
                                           scope=tik.scope_ubuf)
        tik_instance.data_move(input_q_ubuf, input_q[i * 16], 0, 1, shape // 8, 0, 0)
        tik_instance.vec_abs(shape, input_q_ubuf, input_q_ubuf, 1, 0, 0)
        tik_instance.data_move(output[i * 16], input_q_ubuf, 0, 1, shape // 8, 0, 0)

    tik_instance.BuildCCE(kernel_name='FlashAttention', inputs=[input_q], outputs=[output])
    return tik_instance


if __name__ == '__main__':
    tik_instance = flash_attention(None, None)
    data = -numpy.ones((24,), dtype=numpy.float32)
    feed_dict = {'input_q': data}
    # 启动功能调试
    o, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    print(o)
    # 打印输出数据
