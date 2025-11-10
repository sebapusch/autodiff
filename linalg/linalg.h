#ifndef INCLUDED_LINALG
#define INCLUDED_LINALG

#include "../tensor/tensor.h"

namespace autodiff
{
    struct MatmulBroadcastPlan : BroadcastPlan
    {
        size_t rows;
        size_t cols;
        size_t shared;
        size_t max_rank;
        size_t batch_size;
    };

    Tensor matmul(const Tensor &t1, const Tensor &t2);
}

#endif