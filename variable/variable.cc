#include "variable.ih"

namespace autodiff
{
    Variable::Variable(Tensor &&data)
    :
        d_var_data(make_shared<VariableData>(std::move(data)))
    {}

    Variable::Variable(double value)
    :
        d_var_data(make_shared<VariableData>(value))
    {}

    Variable::Variable(Variable const &other)
    :
        d_var_data(other.d_var_data)
    {}

    Variable::Variable(VariableData &&var_data)
    :
        d_var_data(make_shared<VariableData>(std::move(var_data)))
    {}

    Tensor const &Variable::data() const
    {
        return d_var_data->d_data;
    }

    optional<Tensor> const &Variable::grad() const
    {
        return d_var_data->d_grad;
    }

    void Variable::backward()
    {
        if (not data().is_scalar())
            throw runtime_error("Only scalars can back be backpropagated with no incoming gradient");

        auto grad = Tensor{d_var_data->d_data.shape(), 1};

        return backward(grad);
    }

    void Variable::backward(Tensor const &incoming_grad)
    {
        if (not grad())
            d_var_data->d_grad = Tensor{d_var_data->d_data.shape()};

        d_var_data->d_grad.value() += incoming_grad;

        if (d_var_data->d_op)
            d_var_data->d_op->backward(incoming_grad);
    }
}
