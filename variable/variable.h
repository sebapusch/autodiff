#ifndef INCLUDED_VARIABLE
#define INCLUDED_VARIABLE

#include <memory>
#include <optional>

#include "variable_data.h"

namespace autodiff 
{
    class Variable
    {
        std::shared_ptr<VariableData> d_var_data;

        public:
//            Variable(double value);
            Variable(Tensor &&data);
            Variable(Variable const &other);    // keep underlying VariableData

            // getters
            Tensor                  const &data();
            std::optional<Tensor>   const &grad();

        protected:
            Variable(VariableData &&var_data);

        // operations
        friend Variable operator+(Variable const &lhs, Variable const &rhs);
    };
}

#endif
