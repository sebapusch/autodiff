#include "linalg.ih"

namespace autodiff
{
    MatmulBroadcastPlan prepare_matmul_broadcast(Tensor const &lhs, Tensor const &rhs)
    {
        auto const &lhs_shape   = lhs.shape();
        auto const &lhs_strides = lhs.strides();
        auto const &rhs_shape   = rhs.shape();
        auto const &rhs_strides = rhs.strides();

        size_t const rhs_rank = rhs.rank();
        size_t const lhs_rank = lhs.rank();

        size_t const max_rank = max(lhs_rank, rhs_rank);
        size_t const res_rank = lhs_rank == 1 or rhs_rank == 1
            ? max_rank - 1
            : max_rank;

        vector<size_t> b_lhs_strides(max_rank, 1), b_rhs_strides(max_rank, 1),
                       result_shape(res_rank), b_res_strides(max_rank, 1);

        size_t stride_acc, rows, cols, batch_size;
        if (lhs_rank == 1)
        {
            if (rhs_rank == 1)
                throw runtime_error("Incompatible shapes");
            if (rhs_shape[rhs_rank - 2] != lhs_shape[0])
                throw runtime_error("Incompatible shapes");

            b_lhs_strides[max_rank - 2] = 0;
            b_rhs_strides[max_rank - 2] = rhs_strides[rhs_rank - 2];
            b_res_strides[max_rank - 2] = 0;

            result_shape[res_rank - 1] = stride_acc = rhs_shape[rhs_rank - 1];

            rows = 1;
            cols = rhs_shape[rhs_rank - 1];
        }
        else if (rhs_rank == 1)
        {
            if (lhs_shape[lhs_rank - 1] != rhs_shape[0])
                throw runtime_error("Incompatible shapes");

            b_lhs_strides[max_rank - 2] = lhs_strides[lhs_rank - 2];
            b_rhs_strides[max_rank - 1] = 0;
            b_res_strides[max_rank - 1] = 0;

            result_shape[res_rank - 1] = stride_acc = lhs_shape[lhs_rank - 2];

            rows = lhs_shape[lhs_rank - 2];
            cols = 1;
        }
        else
        {
            if (lhs_shape[lhs_rank - 1] != rhs_shape[rhs_rank - 2])
                throw runtime_error("Incompatible shapes");

            b_lhs_strides[max_rank - 2] = lhs_strides[lhs_rank - 2];
            b_rhs_strides[max_rank - 2] = rhs_strides[rhs_rank - 2];
            b_res_strides[max_rank - 2] = rhs_shape[rhs_rank - 1];

            result_shape[res_rank - 2] = lhs_shape[lhs_rank - 2];
            result_shape[res_rank - 1] = rhs_shape[rhs_rank - 1];

            stride_acc = result_shape[res_rank - 2] * result_shape[res_rank - 1];

            rows = lhs_shape[lhs_rank - 2];
            cols = rhs_shape[rhs_rank - 1];
        }

        batch_size = stride_acc;

        int il = static_cast<int>(lhs_rank) - 3;
        int ir = static_cast<int>(rhs_rank) - 3;

        for (int i = static_cast<int>(max_rank) - 3; i >= 0; --i)
        {
            size_t dim_lhs = il >= 0 ? lhs_shape[il]   : 1;
            size_t dim_rhs = ir >= 0 ? rhs_shape[ir]   : 1;

            if (dim_lhs != 1 && dim_rhs != 1 && dim_lhs != dim_rhs)
                throw runtime_error("Incompatible shapes at axis " + to_string(i));

            result_shape[i] = max(dim_lhs, dim_rhs);

            b_lhs_strides[i] = (dim_lhs == 1) ? 0 : lhs_strides[il];
            b_rhs_strides[i] = (dim_rhs == 1) ? 0 : rhs_strides[ir];
            b_res_strides[i] = stride_acc;

            stride_acc *= result_shape[i];

            --il;
            --ir;
        }

        MatmulBroadcastPlan out;

        out.res_strides = b_res_strides;
        out.lhs_strides = b_lhs_strides;
        out.rhs_strides = b_rhs_strides;
        out.batch_size = batch_size;
        out.res_shape = result_shape;
        out.max_rank = max_rank;
        out.rows = rows;
        out.cols = cols;
        out.shared = lhs_shape[lhs_rank - 1];

        return out;
    }

    Tensor matmul(Tensor const &lhs, Tensor const &rhs)
    {
        MatmulBroadcastPlan plan = prepare_matmul_broadcast(lhs, rhs);

        auto const &lhs_data = lhs.cbegin();
        auto const &rhs_data = rhs.cbegin();

        auto const &lhs_strides = plan.lhs_strides;
        auto const &rhs_strides = plan.rhs_strides;
        auto const &res_strides = plan.res_strides;

        vector<double> res(accumulate(plan.res_shape.begin(),
                                        plan.res_shape.end(),
                                        1, multiplies<size_t>()));

        const size_t num_batches = plan.batch_size == 1 ? 1 : res.size() / plan.batch_size;

        for (size_t batch = 0; batch < num_batches; ++batch)
        {
            size_t res_offset = batch * plan.batch_size;
            size_t lhs_offset = 0;
            size_t rhs_offset = 0;

            size_t remaining  = res_offset;
            for (size_t dim = 0; dim < plan.max_rank - 2; ++dim)
            {
                size_t coord = remaining / res_strides[dim];

                lhs_offset += coord * lhs_strides[dim];
                rhs_offset += coord * rhs_strides[dim];

                remaining %= res_strides[dim];
            }

            for (size_t row = 0; row < plan.rows; ++row)
            {
                for (size_t col = 0; col < plan.cols; ++col)
                {
                    size_t i_res = res_offset
                                   + row * res_strides[plan.max_rank - 2]
                                   + col * res_strides[plan.max_rank - 1];
                    double sum = 0;
                    for (size_t shd = 0; shd < plan.shared; ++shd)
                    {
                        size_t i_lhs = lhs_offset
                                       + row * lhs_strides[plan.max_rank - 2]
                                       + shd * lhs_strides[plan.max_rank - 1];
                        size_t i_rhs = rhs_offset
                                       + shd * rhs_strides[plan.max_rank - 2]
                                       + col * rhs_strides[plan.max_rank - 1];

                        sum += lhs_data[i_lhs] * rhs_data[i_rhs];
                    }

                    res[i_res] = sum;
                }
            }

        }

        return Tensor{std::move(plan.res_shape), std::move(res)};
    }


}
