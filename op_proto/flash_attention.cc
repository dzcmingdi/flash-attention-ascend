#include "flash_attention.h"
#include <iostream>
namespace ge {

IMPLEMT_COMMON_INFERFUNC(FlashAttentionInferShape)
{
    TensorDesc tensor_desc_output = op.GetOutputDescByName("o");
    TensorDesc desc_q = op.GetInputDescByName("q");
    Shape q_shape = desc_q.GetShape();
    Shape v_shape = op.GetInputDescByName("v").GetShape();
    Shape k_shape = op.GetInputDescByName("k").GetShape();

    std::vector<int64_t> dims_q = q_shape.GetDims();
    std::vector<int64_t> dims_v = v_shape.GetDims();
    std::vector<int64_t> dims_k = k_shape.GetDims();

    std::vector<int64_t> dims_o;

    dims_o.push_back(dims_q[0]);
    dims_o.push_back(dims_q[1]);
    dims_o.push_back(dims_v[2]);
//    dims_o.push_back(dims_k[2]);


    tensor_desc_output.SetShape(ge::Shape(dims_o));

    tensor_desc_output.SetDataType(desc_q.GetDataType());
    (void) op.UpdateOutputDesc("o", tensor_desc_output);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FlashAttention, FlashAttentionVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FlashAttention, FlashAttentionInferShape);
VERIFY_FUNC_REG(FlashAttention, FlashAttentionVerify);

}  // namespace ge
