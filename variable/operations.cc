#include "variable.ih"

namespace autodiff
{
    Variable operator+(Variable const &lhs, Variable const &rhs)
    {
        auto op = make_shared<Add>();
        auto res = (*op)({lhs, rhs});
        auto var_data = VariableData(std::move(res), op);

        return Variable(std::move(var_data));
    }
    
    Variable operator*(Variable const &lhs, Variable const &rhs)
    {
        auto op = make_shared<Multiply>();
        auto res = (*op)({lhs, rhs});
        auto var_data = VariableData(std::move(res), op);

        return Variable(std::move(var_data));
    }
}
