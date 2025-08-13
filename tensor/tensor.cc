#include "tensor.ih"

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

    Tensor::Tensor(vector<size_t> &&shape, double value)
    :
        d_data(DataPtr(new vector<double>(
            accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>()),
            value
        ))),
        d_strides(calculate_strides(shape)),
        d_shape(std::move(shape)),
        d_length(d_data->size())
    {
        assert(not d_shape.empty() and "shape cannot be empty");
        assert(find(d_shape.begin(), d_shape.end(), 0) == d_shape.end() and "invalid dimension 0");
    }

    Tensor::Tensor(vector<size_t> &&shape)
    :
        Tensor(std::move(shape), 0.0)
    {}

    Tensor::Tensor(std::vector<std::size_t> &&shape, std::vector<double> &&data)
    :
        d_data(DataPtr(new vector<double>(std::move(data)))),
        d_strides(calculate_strides(shape)),
        d_shape(std::move(shape)),
        d_length(d_data->size())
    {
        assert(not d_shape.empty() and "shape cannot be empty");
        assert(find(d_shape.begin(), d_shape.end(), 0) == d_shape.end() and "invalid dimension 0");
        assert(accumulate(d_shape.begin(), d_shape.end(), 1, multiplies<size_t>()) and
            "invalid shape");
    }

    Tensor::Tensor(Tensor const &t, size_t idx)
    :
        d_data(DataPtr(t.d_data)),
        d_strides(calculate_strides(t.d_shape, 1)),
        d_shape(vector<size_t>(t.d_shape.begin() + 1, t.d_shape.end())),
        d_offset(t.d_offset + t.d_strides[0] * idx),
        d_length(t.d_length / t.d_shape[0])
    {
        assert(t.rank() != 1 and "cannot index one-dimensional tensor");
    }

    Tensor::~Tensor()
    {}


    Tensor Tensor::operator[](size_t idx)
    {
        if (rank() == 1)
            throw runtime_error("cannot index one dimensional array");

        if (idx >= d_shape[0])
            throw_out_of_bound_error(0, d_shape[0] - 1, idx);

        return Tensor{*this, idx};
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
