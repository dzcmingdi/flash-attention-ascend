from tbe import tik
from flash_attention import transpose_left_matrix, row_sum
import tbe.common.platform as tbe_platform
import numpy


class Test:

    def kernel0(self, q_type, kernel_name="kernel0"):
        q_shape = q_type['shape']
        k0 = 16
        k = q_shape[1]
        k1 = (k + k0 - 1) // k0
        m = 16

        tik_instance = tik.Tik(disable_debug=False)
        q = tik_instance.Tensor(dtype='float16', shape=q_type['shape'], name="q", scope=tik.scope_gm)
        q_ = tik_instance.Tensor(dtype='float16', shape=[k1, m, k0], name="q_", scope=tik.scope_gm)
        transpose_left_matrix(tik_instance, q, q_, 'float16', k1, m, k0)

        tik_instance.BuildCCE(kernel_name="kernel0", inputs=[q], outputs=[q_])
        return tik_instance

    def kernel1(self, q_type, kernel_name="kernel1"):
        q_shape = q_type['shape']
        tik_instance = tik.Tik(disable_debug=False)
        q = tik_instance.Tensor(dtype='float16', shape=q_shape, name="q", scope=tik.scope_gm)
        q_ = tik_instance.Tensor(dtype='float16', shape=q_shape, name="q", scope=tik.scope_ubuf)
        tik_instance.data_move(q_, q, 0, 1, (q_shape[0] * q_shape[1] * q_shape[2]) // 16, 0, 0)
        _output = tik_instance.Tensor(dtype='float16', shape=[q_shape[1]], name='output', scope=tik.scope_ubuf)
        row_sum(q_, _output, tik_instance)

        output = tik_instance.Tensor(dtype='float16', shape=[q_shape[1]], name='output', scope=tik.scope_gm)

        tik_instance.data_move(output, _output, 0, 1, q_shape[1] // 16, 0, 0)
        tik_instance.BuildCCE(kernel_name="kernel1", inputs=[q], outputs=[output])

        return tik_instance


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


def test0(t_: Test):
    a = numpy.random.random((16, 32)).astype(numpy.float16)

    print(a.reshape((16, 2, 16)).transpose((1, 0, 2)))
    tik_instance = t_.kernel0({'shape': a.shape})
    o, = tik_instance.tikdb.start_debug(feed_dict={'q': a}, interactive=False)
    assert (a != o).sum() == 0.


def test1(t_: Test):
    a = numpy.random.random((2, 16, 16)).astype(numpy.float16)
    print(a.sum(axis=0).sum(axis=1))
    tik_instance = t_.kernel1({'shape': a.shape})
    o, = tik_instance.tikdb.start_debug(feed_dict={'q': a}, interactive=False)
    print(o)


def test2(t_: Test):
    a = numpy.zeros((2, 16, 16), dtype=numpy.float16)

    a[:, 1, :] = 2.
    tik_instance = tik.Tik(disable_debug=False)

    q = tik_instance.Tensor(dtype='float16', shape=[2, 16, 16], name="q", scope=tik.scope_gm)
    q_ = tik_instance.Tensor(dtype='float16', shape=[2, 16, 16], name="q_", scope=tik.scope_ubuf)
    scalar = tik_instance.Scalar(dtype='float16', name='scalar', init_value=2.)
    tik_instance.vec_dup(16, q_[1 * 16], scalar, 2, 16)

    o = tik_instance.Tensor(dtype='float16', shape=[2, 16, 16], name='o', scope=tik.scope_gm)

    tik_instance.data_move(o, q_, 0, 1, 32, 0, 0)

    tik_instance.BuildCCE(kernel_name="test2", inputs=[q], outputs=[o])

    o, = tik_instance.tikdb.start_debug(feed_dict={'q': a}, interactive=False)

    print(a)

    print(o)


if __name__ == '__main__':
    t = Test()
    test1(t)
