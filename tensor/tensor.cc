#include "tensor.ih"
#include <iomanip>

namespace autodiff
{
    Tensor::Tensor(vector<size_t> &&shape, double value)
    :
        d_shape(shape),
        d_strides(calculate_strides(shape)),
        d_data(DataPtr(new vector<double>(
            accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>()),
            // @todo: what if shape is {}?
            value
        ))),
        d_start(d_data->begin()),
        d_end(d_data->end())
    {}

    Tensor::Tensor(vector<size_t> &&shape)
    :
        Tensor(move(shape), 0.0)
    {}

    // @todo: fix double initialization of data.
    Tensor::Tensor(std::vector<std::size_t> &&shape, std::vector<double> &&data)
    :
        Tensor(move(shape))
    {
        size_t size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
        if (size != data.size())
            // @todo: better error message
            throw runtime_error("Invalid shape provided");

        d_data  = DataPtr(new vector<double>(move(data)));
        d_start = d_data->begin();
        d_end   = d_start + size;
    }

    /* @todo: ensure index is compatible */
    Tensor::Tensor(Tensor const &t, size_t idx)
    :
        Tensor(vector<size_t>(t.shape().begin() + 1, t.shape().end()))
    {
        d_strides = vector<size_t>(t.d_strides.begin() + 1, t.d_strides.end());
        d_data  = t.d_data;
        d_start = d_data->begin() + t.d_strides[0] * idx;
        d_end   = d_data->begin() + t.d_strides[0] * (idx + 1);
    }

    Tensor Tensor::operator[](size_t idx)
    {
        if (d_shape.size() == 0)
            throw runtime_error("Cannot index tensor of shape (0,)");

        if (idx >= d_shape[0])
            throw err_out_of_bound(0, d_shape[0] - 1, idx);

        return Tensor{*this, idx};
    }

    Tensor::~Tensor()
    {

    }

/*
    Tensor Tensor::operator[](size_t start, size_t end)
    {
        if (d_shape.size() == 0)
            throw runtime_error("Cannot index tensor of shape (0,)");

        if (start > end)
            throw runtime_error("Start index must be lte end index");

        // @todo: additional checks

        return TensorView{*this, {tuple(start, end)}};
    }
*/
    //--------------------
    // ACCESSORS
    //--------------------

    vector<double> const &Tensor::data() const
    {
        return *d_data;
    }

    vector<size_t> const &Tensor::shape() const
    {
        return d_shape;
    }

    vector<size_t> const &Tensor::strides() const
    {
        return d_strides;
    }

    Tensor::DataStartConst Tensor::begin() const
    {
        return DataStartConst(d_start);
    }

    Tensor::DataStartConst Tensor::end() const
    {
        return DataStartConst(d_end);
    }

    size_t Tensor::rank() const
    {
        return d_shape.size();
    }

    ostream &operator<<(ostream &out, Tensor const &t)
    {
        //auto const &shape   = t.shape();
        //auto const &strides = t.strides();
        auto const &data    = t.data();

        size_t const size = data.size();
        size_t const rank = t.rank();
        for (size_t i = 0; i < size; ++i)
        {
            out << data[i] << ", ";
            for (size_t axis = 0; axis < rank; ++axis)
            {
                //if (i % strides[axis])
                //    out << "\n";
            }
        }

        out << "\n";

        return out;
    }
}

runtime_error err_out_of_bound(size_t dim, size_t max, size_t idx)
{
    return runtime_error("Index out of bounds on dimension "
        + to_string(dim) + ":"
        + to_string(max) + ", "
        + to_string(idx) + " received.");
}

vector<size_t> calculate_strides(const vector<size_t> &shape)
{
    size_t size = shape.size();
    std::vector<size_t> strides(size);

    size_t stride = 1;
    for (size_t i = size; i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}
