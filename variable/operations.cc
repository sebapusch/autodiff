#include "variable.ih"

namespace autodiff
{
    Variable operator+(Variable const &lhs, Variable const &rhs)
    {
        auto op = make_shared<Sum>();
        auto res = (*op)({lhs, rhs});
        auto var_data = VariableData(std::move(res), op);

        return Variable(std::move(var_data));
    }
}
