from tbe import tik
import tbe.common.platform as tbe_platform
import numpy


def test(kernel_name="test"):
    tik_instance = tik.Tik(disable_debug=False)
    t_ = tik_instance.Tensor(dtype='float16', shape=[16, 16], name='t', scope=tik.scope_gm)
    t_ub = tik_instance.Tensor(dtype='float16', shape=[16, 16], name='t_ub', scope=tik.scope_ubuf)
    tik_instance.data_move(t_ub, t_, 0, 1, 16, 0, 0)
    dst_gm = tik_instance.Tensor('float16', [16, 16], name="dst_gm", scope=tik.scope_gm)
    dst_ub = tik_instance.Tensor('float16', [16, 16], name="dst_ub", scope=tik.scope_ubuf)
    dst_list = [dst_ub[16 * i] for i in range(16)]
    src_list = [t_ub[16 * i] for i in range(16)]
    dst_rep_stride = 0
    src_rep_stride = 0
    tik_instance.vec_trans_scatter(False, False, dst_list, src_list, 1, dst_rep_stride, src_rep_stride)
    tik_instance.data_move(dst_gm, dst_ub, 0, 1, 16, 0, 0)

    tik_instance.BuildCCE(kernel_name="test", inputs=[t_], outputs=[dst_gm])

    return tik_instance
