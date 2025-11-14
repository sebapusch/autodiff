#ifndef INCLUDED_TENSOR
#define INCLUDED_TENSOR

#include <cstddef>
#include <tuple>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

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
        Tensor(Tensor const &t) = default;

        Tensor(double value);                   // array value (empty shape)
        Tensor(std::vector<size_t> &&shape);
        Tensor(std::vector<size_t> const &shape);
        Tensor(std::vector<size_t> &&shape, double value);
        Tensor(std::vector<size_t> const &shape, double value);
        Tensor(std::vector<size_t> &&shape, std::vector<double> &&data);
        Tensor(std::vector<size_t> const &shape, std::vector<double> &&data);
        ~Tensor();

    private:
        Tensor(
            std::vector<size_t> &&shape,
            std::vector<size_t> &&strides,
            DataPtr data,
            size_t  offset,
            size_t  length);

    public:

        std::vector<size_t> const &shape()      const;
        std::vector<size_t> const &strides()    const;

        size_t rank() const;
        size_t size() const;

        double scalar() const;
        double &scalar();

        DataConstIter cbegin()   const;
        DataConstIter cend()     const;

        Tensor operator[](size_t idx) &;
        Tensor operator[](size_t idx) &&;
        Tensor operator()(size_t idx1, size_t idx2, ...);

        Tensor &operator=(double value);
        Tensor &operator=(Tensor &&t) & = default;
        Tensor &operator=(Tensor &&t) &&;
        Tensor &operator=(Tensor &t) &&;

        Tensor &operator+=(Tensor const &rhs);
        Tensor &operator+=(double number);

        Tensor &operator-=(Tensor const &rhs);
        Tensor &operator-=(double number);

        Tensor &operator*=(Tensor const &rhs);
        Tensor &operator*=(double number);

        Tensor &operator/=(Tensor const &rhs);
        Tensor &operator/=(double number);

        double sum() const;

        // -- check
        Tensor &power(double num);

    private:
        DataIter begin();
        DataIter end();

        void compatible(Tensor const &other) const;

        friend void swap(Tensor& a, Tensor& b) noexcept;
    };

    std::ostream &operator<<(std::ostream &out, autodiff::Tensor const &t);

    autodiff::Tensor operation(autodiff::Tensor const &a,
                               autodiff::Tensor const &b,
                               std::function<double(double, double)> op);

    // --- arithmetic.cc
    Tensor operator+(Tensor const &lhs, Tensor const &rhs);
    Tensor operator+(Tensor const &lhs, double rhs);

    Tensor operator-(Tensor const &lhs, Tensor const &rhs);
    Tensor operator-(Tensor const &lhs, double rhs);

    Tensor operator*(Tensor const &lhs, Tensor const &rhs);
    Tensor operator*(Tensor const &lhs, double rhs);

    Tensor operator/(Tensor const &lhs, Tensor const &rhs);
    Tensor operator/(Tensor const &lhs, double rhs);
    // /-- arithmetic.cc

    // --- ops.cc
    Tensor maximum(Tensor const &lhs, Tensor const &rhs);
    Tensor concatenate(Tensor const &lhs, Tensor const &rhs, std::optional<size_t> axis = 0);
    // /-- ops.cc

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
