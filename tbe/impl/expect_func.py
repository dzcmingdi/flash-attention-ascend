import numpy


def expect(q, k, v):
    results = []
    for bq, bk, bv in zip(q, k, v):
        s = numpy.matmul(bq, bk)
        m = numpy.max(s, axis=1, keepdims=True)
        p = numpy.exp(s - m)
        l = numpy.sum(p, axis=1, keepdims=True)
        scores = p / l
        results.append(numpy.matmul(scores, bv))

    return numpy.stack(results)
