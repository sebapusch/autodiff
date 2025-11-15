#     include "variable.ih"

namespace autodiff
{

    VariableData::VariableData(Tensor &&data)
    :
        d_data(move(data)),
        d_grad({}),
        d_op(nullptr)
    {}

    VariableData::VariableData(Tensor &&data, shared_ptr<Operator> op)
    : VariableData(move(data))
    {
        d_op = std::move(op);
    }

    VariableData::VariableData(double val)
    : VariableData(Tensor(val))
    {}
    
    VariableData::VariableData(double val, shared_ptr<Operator> op)
    : VariableData(Tensor(val), op)
    {}
}
