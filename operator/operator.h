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
            
            void backward(Tensor const &incoming_grad);

        protected:
            virtual Tensor  compute_forward() = 0;
            virtual void    compute_backward(Tensor const &incoming_grad) = 0;
    };

    class Add : public Operator
    {
        protected:
            Tensor  compute_forward() override;
            void    compute_backward(Tensor const &incoming_grad) override;
    };

    class Multiply : public Operator
    {
        protected:
            Tensor  compute_forward() override;
            void    compute_backward(Tensor const &incoming_grad) override;
    };
}

#endif
