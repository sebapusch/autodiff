#include "tensor.ih"
#include "tensor.h"
#include <stdexcept>

namespace autodiff
{
    namespace
    {
        void throw_out_of_bound_error(size_t dim, size_t max, size_t idx)
        {
            string error_msg = "Index out of bounds on dimension "
                + to_string(dim) + ":"
                + to_string(max) + ", "
                + to_string(idx) + " received.";

            throw invalid_argument(error_msg);
        }

        vector<size_t> calculate_strides(vector<size_t> const &shape, size_t start_ix = 0)
        {
            size_t size = shape.size() - start_ix;
            vector<size_t> strides(size);

            size_t acc = 1;
            for (size_t dim = size; dim-- > 0;)
            {
                strides[dim] = acc;
                acc *= shape[start_ix + dim];
            }

            return strides;
        }
    }

    Tensor::Tensor(double value)
    :
        d_data(make_shared<vector<double>>(1, value)),
        d_strides({1}),
        d_shape({1}),
        d_length(1)
    {}

    Tensor::Tensor(vector<size_t> &&shape, double value)
    :
        d_data(make_shared<vector<double>>(
            accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>()),
            value
        )),
        d_strides(calculate_strides(shape)),
        d_shape(std::move(shape)),
        d_length(d_data->size())
    {
        assert(not d_shape.empty() and "shape cannot be empty");
        assert(find(d_shape.begin(), d_shape.end(), 0) == d_shape.end() and "invalid dimension 0");
    }

    Tensor::Tensor(vector<size_t> const &shape, double value)
    :
        Tensor(vector<size_t>(shape), value)
    {}

    Tensor::Tensor(vector<size_t> &&shape)
    :
        Tensor(std::move(shape), 0.0)
    {}

    Tensor::Tensor(vector<size_t> const &shape)
    :
        Tensor(vector<size_t>(shape))
    {}

    Tensor::Tensor(vector<size_t> &&shape, vector<double> &&data)
    :
        d_data(make_shared<vector<double>>(std::move(data))),
        d_strides(calculate_strides(shape)),
        d_shape(std::move(shape)),
        d_length(d_data->size())
    {
        assert(not d_shape.empty() and "shape cannot be empty");
        assert(find(d_shape.begin(), d_shape.end(), 0) == d_shape.end() and "invalid dimension 0");
        assert(accumulate(d_shape.begin(), d_shape.end(), 1, multiplies<size_t>()) and
            "invalid shape");
    }

    Tensor::Tensor(vector<size_t> const &shape, vector<double> &&data)
    :
        Tensor(vector<size_t>(shape), std::move(data))
    {}

    Tensor::Tensor(
        vector<size_t> &&shape,
        vector<size_t> &&strides,
        DataPtr data,
        size_t offset,
        size_t length)
    :
        d_data(data),
        d_strides(strides),
        d_shape(shape),
        d_offset(offset),
        d_length(length)
    {}

    Tensor::~Tensor()
    {}

    Tensor Tensor::operator[](size_t idx) &
    {
        assert(rank() > 0 and "cannot index tensor of rank 0 (scalar)");

        if (idx >= d_shape[0])
            throw_out_of_bound_error(0, d_shape[0] - 1, idx);

        return Tensor{
            vector<size_t>(d_shape.begin() + 1, d_shape.end()),
            vector<size_t>(d_strides.begin() + 1, d_strides.end()),
            d_data,
            d_offset + d_strides[0] * idx,
            d_length / d_shape[0]
        };
    }

    Tensor Tensor::operator[](size_t idx) &&
    {
        return (*this)[idx];
    }

    Tensor &Tensor::operator=(double val)
    {
        for_each(begin(), end(), [val](double &old) {
            old = val;
        });

        return *this;
    }

    // @todo maybe try to remove duplicate
    Tensor &Tensor::operator=(Tensor &t) &&
    {
        if (rank() != t.rank())
            throw invalid_argument("rank must match");

        size_t ix = 0;
        for_each(d_shape.begin(), d_shape.end(), [&ix, &t](size_t dim) {
            if (t.d_shape[ix++] != dim) throw invalid_argument("incompatible shape");
        });

        copy(t.cbegin(), t.cend(), begin());

        return *this;
    }

    Tensor &Tensor::operator=(Tensor &&t) &&
    {
        if (rank() != t.rank())
            throw invalid_argument("rank must match");

        size_t ix = 0;
        for_each(d_shape.begin(), d_shape.end(), [&ix, &t](size_t dim) {
            if (t.d_shape[ix++] != dim) throw invalid_argument("incompatible shape");
        });

        copy(t.cbegin(), t.cend(), begin());

        return *this;
    }

    vector<size_t> const &Tensor::shape() const
    {
        return d_shape;
    }

    vector<size_t> const &Tensor::strides() const
    {
        return d_strides;
    }

    size_t Tensor::rank() const
    {
        return d_shape.size();
    }

    size_t Tensor::size() const
    {
        return d_length;
    }

    bool Tensor::is_scalar() const
    {
        return rank() == 1 and d_shape[0] == 1;
    }

    double &Tensor::scalar()
    {
        assert(is_scalar() and "only tensors of rank 0 (scalar) can be converted to scalar");
        return *begin();
    }

    double Tensor::scalar() const
    {
        assert(is_scalar() and "only tensors of rank 0 (scalar) can be converted to scalar");
        return *cbegin();
    }

    Tensor::DataConstIter Tensor::cbegin() const
    {
        return d_data->cbegin() + d_offset;
    }

    Tensor::DataConstIter Tensor::cend() const
    {
        return d_data->cbegin() + d_offset + d_length;
    }

    Tensor::DataIter Tensor::begin()
    {
        return d_data->begin() + d_offset;
    }

    Tensor::DataIter Tensor::end()
    {
        return d_data->begin() + d_offset + d_length;
    }

    void swap(Tensor& a, Tensor& b) noexcept
    {
        std::swap(a.d_data,    b.d_data);
        std::swap(a.d_shape,   b.d_shape);
        std::swap(a.d_strides, b.d_strides);
        std::swap(a.d_offset,  b.d_offset);
        std::swap(a.d_length,  b.d_length);
    }

    Tensor operation(Tensor const &lhs, Tensor const &rhs, function<double(double, double)> operator_)
    {
        BroadcastPlan plan = prepare_broadcast(lhs, rhs);

        auto const &lhs_strides = plan.lhs_strides;
        auto const &rhs_strides = plan.rhs_strides;
        auto const &res_strides = plan.res_strides;

        auto &res_shape   = plan.res_shape;

        auto const &lhs_data = lhs.cbegin();
        auto const &rhs_data = rhs.cbegin();

        size_t rank = res_shape.size();
        size_t res_size = accumulate(res_shape.begin(), res_shape.end(), 1, multiplies<size_t>());

        vector<double> res(res_size);

        size_t i_lhs = 0;
        size_t i_rhs = 0;

        for (size_t i = 0; i < res_size; ++i)
        {
            i_lhs = 0;
            i_rhs = 0;

            size_t j = i;
            for (size_t axis = 0; axis < rank; ++axis)
            {
                size_t coord = j / res_strides[axis];
                j = j % res_strides[axis];

                i_lhs += coord * lhs_strides[axis];
                i_rhs += coord * rhs_strides[axis];
            }

            res[i] = operator_(lhs_data[i_lhs], rhs_data[i_rhs]);
        }

        return Tensor{std::move(res_shape), std::move(res)};
    }

    ostream &operator<<(ostream &out, Tensor const &t)
    {
        out << "(";
        for (size_t dim = 0; dim < t.rank(); ++dim)
        {
            out << t.shape()[dim];
            if (dim < t.rank() - 1)
                out << ", ";
        }

        out << ")\n[";
        if (t.rank() > 1) out << "\n";

        size_t ix = 0;
        for_each(t.cbegin(), t.cend(), [&ix, &t, &out](double val) {
            size_t rem = ix;
            string open, close;
            for (size_t dim = 0; dim < t.rank() - 1; ++dim)
            {
                size_t dim_inv = t.rank() - 2 - dim;
                if (ix > 0 and rem % t.strides()[dim_inv] == 0)
                {
                    if (dim_inv != t.rank() - 2)
                        close += string(3 * (dim_inv + 1), ' ');
                    close += "]\n";
                }

                if (rem % t.strides()[dim] == 0)
                {
                    open += string(3 * (dim + 1), ' ') + "[";
                    if (dim != t.rank() - 2)
                        open += "\n";
                }

                rem %= t.strides()[dim];
            }

            out << close << open << val;
            if ((rem + 1) % t.shape()[t.rank() - 1] != 0)
                out << ", ";

            ++ix;
        });

        out << "]\n";
        for (size_t dim = t.rank() - 1; dim-- > 0;)
            out << string(3 * dim, ' ') + "]\n";

        return out;
    }
}
