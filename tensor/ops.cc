#include "tensor.ih"

namespace autodiff
{
    namespace
    {
        void throw_concatenation_dim_mismatch_error(size_t dim, size_t lhs_shape, size_t rhs_shape)
        {
            string error_msg = "all the input array dimensions except for the concatenation axis must "
                                "match exactly, but along dimension " + to_string(dim) +
                                ", the array at index 0 has size " + to_string(lhs_shape) +
                                " and the array at index 1 has size " + to_string(rhs_shape);
            throw runtime_error(error_msg);
        }

        void throw_rank_mismatch_error(size_t lhs_rank, size_t rhs_rank) {
            string error_msg = "all the input arrays must have same number of dimensions, "
                                "the array at index 0 has " + to_string(lhs_rank) + " dimension(s), "
                                "the array at index 1 has " + to_string(rhs_rank) + " dimension(s)";
            throw runtime_error(error_msg);
        }
    }

    Tensor concatenate(Tensor const &lhs, Tensor const &rhs, optional<size_t> axis)
    {
        if (lhs.rank() != rhs.rank())
            throw_rank_mismatch_error(lhs.rank(), rhs.rank());

        auto const &lhs_shape = lhs.shape();
        auto const &rhs_shape = rhs.shape();

        vector<size_t> res_shape;

        if (axis.has_value())
        {
            res_shape.reserve(lhs.rank());
            for (size_t dim = 0; dim < lhs.rank(); ++dim)
            {
                if (dim == axis.value())
                    res_shape.push_back(lhs_shape[dim] + rhs_shape[dim]);
                else
                {
                    if (lhs_shape[dim] != rhs_shape[dim])
                        throw_concatenation_dim_mismatch_error(dim, lhs_shape[dim], rhs_shape[dim]);

                    res_shape.push_back(lhs.shape()[dim]);
                }
            }
        }
        else
            res_shape = {lhs.size() + rhs.size()};

        vector<double> res(lhs.size() + rhs.size());

        copy(lhs.cbegin(), lhs.cend(), res.begin());
        copy(rhs.cbegin(), rhs.cend(), res.begin() + lhs.size());

        return Tensor{move(res_shape), move(res)};
    }
}
