#ifndef INCLUDED_TENSOR
#define INCLUDED_TENSOR

#include <cstddef>
#include <tuple>
#include <vector>
#include <memory>
#include <optional>

namespace autodiff
{
    class Tensor
    {
        using DataPtr       = std::shared_ptr<std::vector<double>>;
        using DataIter      = std::vector<double>::iterator;
        using DataConstIter = std::vector<double>::const_iterator;

        DataPtr             d_data;
        std::vector<size_t> d_strides;
        std::vector<size_t> d_shape;
        size_t              d_offset = 0;
        size_t              d_length;

    public:
        Tensor(std::vector<size_t> &&shape);
        Tensor(std::vector<size_t> &&shape, double value);
        Tensor(std::vector<size_t> &&shape, std::vector<double> &&data);
        ~Tensor();

        std::vector<size_t> const &shape()      const;
        std::vector<size_t> const &strides()    const;

        size_t rank() const;
        size_t size() const;

        DataConstIter cbegin()   const;
        DataConstIter cend()     const;

        Tensor operator[](size_t idx);

        Tensor &operator+=(Tensor const &rhs);
        Tensor &operator+=(double number);

        Tensor &operator-=(Tensor const &rhs);
        Tensor &operator-=(double number);

        Tensor &operator*=(Tensor const &rhs);
        Tensor &operator*=(double number);

        Tensor &operator/=(Tensor const &rhs);
        Tensor &operator/=(double number);

        operator double();

    private:
        Tensor(Tensor const &t, size_t idx);

        DataIter begin();
        DataIter end();

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

    Tensor concatenate(Tensor const &lhs, Tensor const &rhs, std::optional<size_t> axis = 0);

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
