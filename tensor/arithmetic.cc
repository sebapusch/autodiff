#include "tensor.ih"

namespace autodiff
{
    namespace
    {

    }

    Tensor operator+(Tensor const &lhs, Tensor const &rhs)
    {
        return operation(lhs, rhs, [](double x, double y) { return x + y; });
    }

    Tensor &Tensor::operator+=(Tensor const &rhs)
    {
        Tensor res = *this + rhs;
        // @todo: might implement custom swap
        swap(*this, res);
        return *this;
    }

    Tensor operator-(Tensor const &lhs, Tensor const &rhs)
    {
        return operation(lhs, rhs, [](double x, double y) { return x - y; });
    }

    Tensor &Tensor::operator-=(Tensor const &rhs)
    {
        Tensor res = *this - rhs;
        swap(*this, res);
        return *this;
    }

    Tensor operator*(Tensor const &lhs, Tensor const &rhs)
    {
        return operation(lhs, rhs, [](double x, double y) { return x * y; });
    }

    Tensor &Tensor::operator*=(Tensor const &rhs)
    {
        Tensor res = *this - rhs;
        swap(*this, res);
        return *this;
    }

    Tensor operator/(Tensor const &lhs, Tensor const &rhs)
    {
        return operation(lhs, rhs, [](double x, double y) { return x / y; });
    }

    Tensor &Tensor::operator/=(Tensor const &rhs)
    {
        Tensor res = *this - rhs;
        swap(*this, res);
        return *this;
    }

    Tensor &Tensor::operator+=(double num)
    {
        for_each(begin(), end(), [num](double &val) {
            val += num;
        });

        return *this;
    }

    Tensor &Tensor::operator-=(double num)
    {
        for_each(begin(), end(), [num](double &val) {
            val -= num;
        });

        return *this;
    }

    Tensor &Tensor::operator*=(double num)
    {
        for_each(begin(), end(), [num](double &val) {
            val *= num;
        });

        return *this;
    }

    Tensor &Tensor::operator/=(double num)
    {
        for_each(begin(), end(), [num](double &val) {
            val /= num;
        });

        return *this;
    }

    Tensor operator*(Tensor const &t, double num)
    {
        vector<double> res(t.size());
        size_t ix = 0;
        for_each(t.cbegin(), t.cend(), [&ix, &res, num](double val) {
            res[ix++] = val * num;
        });

        return Tensor{t.shape(), std::move(res)};
    }

    Tensor &Tensor::power(double num)
    {
        for_each(begin(), end(), [num](double &val) {
            val = std::pow(val, num);
        });

        return *this;
    }

    double Tensor::sum() const
    {
        return accumulate(cbegin(), cend(), 0, plus<double>());
    }

    BroadcastPlan prepare_broadcast(const Tensor& lhs, const Tensor& rhs)
    {
        auto const &lhs_shape   = lhs.shape();
        auto const &lhs_strides = lhs.strides();
        auto const &rhs_shape   = rhs.shape();
        auto const &rhs_strides = rhs.strides();

        const size_t rank = max(lhs_shape.size(), rhs_strides.size());

        vector<size_t>  b_lhs_strides(rank), b_rhs_strides(rank),
                        result_shape(rank), result_strides(rank);

        int ia = static_cast<int>(lhs_shape.size()) - 1;
        int ib = static_cast<int>(rhs_shape.size()) - 1;

        size_t stride_acc = 1;

        for (int i = static_cast<int>(rank) - 1; i >= 0; --i)
        {
            const size_t dimA = ia >= 0 ? lhs_shape[ia]   : 1;
            const size_t dimB = ib >= 0 ? rhs_shape[ib]   : 1;
            const size_t rawA = ia >= 0 ? lhs_strides[ia] : 0;
            const size_t rawB = ib >= 0 ? rhs_strides[ib] : 0;

            if (dimA != 1 && dimB != 1 && dimA != dimB)
                throw runtime_error("Incompatible shapes at axis " + to_string(i));

            const size_t dimR = max(dimA, dimB);
            result_shape[i] = dimR;

            b_lhs_strides[i] = (dimA == 1) ? 0 : rawA;
            b_rhs_strides[i] = (dimB == 1) ? 0 : rawB;

            result_strides[i] = stride_acc;
            stride_acc *= dimR;

            --ia;
            --ib;
        }

        BroadcastPlan out;
        out.lhs_strides = b_lhs_strides;
        out.rhs_strides = b_rhs_strides;
        out.res_strides = result_strides;
        out.res_shape = result_shape;
        return out;
    }
}
