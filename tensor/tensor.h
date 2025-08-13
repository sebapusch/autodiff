#ifndef INCLUDED_TENSOR
#define INCLUDED_TENSOR

#include <cstddef>
#include <tuple>
#include <vector>
#include <memory>

namespace autodiff
{
    class Tensor
    {
        using DataPtr = std::shared_ptr<std::vector<double>>;
        using DataIter = std::vector<double>::iterator;
        using DataStartConst = std::vector<double>::const_iterator;
        using Shape = std::vector<std::size_t>;

        Shape       d_shape;
        Shape       d_strides;
        DataPtr     d_data;
        DataIter    d_start;         // iterator to beginning of data
        DataIter    d_end;           // iterator to end of data

    public:
        Tensor(std::vector<std::size_t> &&shape);
        Tensor(std::vector<std::size_t> &&shape, double value);
        Tensor(std::vector<std::size_t> &&shape, std::vector<double> &&data);
        Tensor(Tensor const &t, size_t idx);
        ~Tensor();

        Tensor operator[](size_t idx);
        Tensor operator[](size_t start, size_t end);

        Tensor &operator+=(Tensor const &rhs);
        Tensor &operator+=(double number);

        Tensor &operator-=(Tensor const &rhs);
        Tensor &operator-=(double number);

        Tensor &operator*=(Tensor const &rhs);
        Tensor &operator*=(double number);

        Tensor &operator/=(Tensor const &rhs);
        Tensor &operator/=(double number);

        std::vector<double> const &data()  const;
        std::vector<size_t> const &shape() const;
        std::vector<size_t> const &strides() const;

        size_t rank() const;

        std::vector<double>::const_iterator begin() const;
        std::vector<double>::const_iterator end() const;

        operator double();

    protected:
        void compatible(Tensor const &other) const;
    };

    std::ostream &operator<<(std::ostream &out, autodiff::Tensor const &t);

    Tensor operator+(Tensor const &lhs, Tensor const &rhs);
    Tensor operator+(Tensor const &lhs, double rhs);

    Tensor operator-(Tensor const &lhs, Tensor const &rhs);
    Tensor operator-(Tensor const &lhs, double rhs);

    Tensor operator*(Tensor const &lhs, Tensor const &rhs);
    Tensor operator*(Tensor const &lhs, double rhs);

    Tensor operator/(Tensor const &lhs, Tensor const &rhs);
    Tensor operator/(Tensor const &lhs, double rhs);

    struct BroadcastPlan
    {
        std::vector<size_t> lhs_strides;
        std::vector<size_t> rhs_strides;
        std::vector<size_t> res_strides;
        std::vector<size_t> res_shape;
    };

    BroadcastPlan prepare_broadcast(Tensor const &lhs, Tensor const &rhs);
}


#endif
