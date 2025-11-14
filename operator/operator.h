#ifndef INCLUDED_OPERATOR
#define INCLUDED_OPERATOR

#include <vector>
#include <initializer_list>

#include "../variable/variable.h"

namespace autodiff
{
    class Operator
    {
        protected:
            std::vector<Variable> d_inputs;
        
        public:
            Operator();

            Tensor operator()(std::initializer_list<Variable> inputs);

        protected:
            virtual Tensor forward() = 0;
    };

    class Sum : public Operator
    {
        protected:
            Tensor forward() override;
    };

}

#endif
