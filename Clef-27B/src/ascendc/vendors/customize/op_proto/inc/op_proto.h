#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ClefGdnQkvNorm)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(q, ge::TensorType::ALL())
    .OUTPUT(k, ge::TensorType::ALL())
    .OUTPUT(v, ge::TensorType::ALL())
    .REQUIRED_ATTR(eps, Float)
    .OP_END_FACTORY_REG(ClefGdnQkvNorm);

}

#endif
