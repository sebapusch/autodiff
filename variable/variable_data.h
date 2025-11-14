#ifndef INCLUDED_VARIABLE_DATA
#define INCLUDED_VARIABLE_DATA

#include <memory>
#include <optional>

//@todo fix this mess
namespace autodiff
{
    class Variable;
    class Operator;
}

#include "../tensor/tensor.h"
#include "../operator/operator.h"

namespace autodiff
{
    class VariableData
    {
        Tensor                      d_data;
        std::optional<Tensor>       d_grad;
        std::shared_ptr<Operator>   d_op = nullptr;

        public:
//            VariableData(double val);
//            VariableData(double val, Operator &&op);

            VariableData(Tensor &&data);
            VariableData(Tensor &&data, std::shared_ptr<Operator> op);

        friend class Variable;
    };
}

#endif

