#include "operator.ih"

namespace autodiff
{
    Operator::Operator()
    {}

    Tensor Operator::operator()(initializer_list<Variable> inputs)
    {
        d_inputs = vector<Variable>{inputs.begin(), inputs.end()}; 

        return forward();
    }

    ///////////////
    // Sum
    //////////////
    Tensor Sum::forward()
    {
        assert(d_inputs.size() == 2 and 
               "Invalid number of inputs, expected 2 but received " + d_inputs.size());

        return d_inputs[0].data() + d_inputs[1].data();
    }

}
