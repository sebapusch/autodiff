#include "variable.ih"

namespace autodiff
{
    Variable::Variable(Tensor &&data)
    :
        d_var_data(make_shared<VariableData>(std::move(data)))
    {}

//    Variable::Variable(double value)
//    :
//        d_var_data(make_shared<VariableData>(value))
//    {}

    Variable::Variable(Variable const &other)
    :
        d_var_data(other.d_var_data)
    {}

    Variable::Variable(VariableData &&var_data)
    :
        d_var_data(make_shared<VariableData>(std::move(var_data)))
    {}

    Tensor const &Variable::data()
    {
        return d_var_data->d_data;
    }

    optional<Tensor> const &Variable::grad()
    {
        return d_var_data->d_grad;
    }
}
