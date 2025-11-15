#include "operator.ih"

namespace autodiff
{
    Operator::Operator()
    {}

    Tensor Operator::operator()(initializer_list<Variable> inputs)
    {
        d_inputs = vector<Variable>{inputs.begin(), inputs.end()}; 

        return compute_forward();
    }

    void Operator::backward(Tensor const &incoming_grad)
    {
        assert(not d_inputs.empty() and "operator not applied");

        compute_backward(incoming_grad);
    }

    ///////////////
    // Add
    //////////////
    Tensor Add::compute_forward()
    {
        assert(d_inputs.size() == 2 and 
               "Invalid number of inputs, expected 2 but received " + d_inputs.size());

        return d_inputs[0].data() + d_inputs[1].data();
    }

    void Add::compute_backward(Tensor const &incoming_grad)
    {
        d_inputs[0].backward(incoming_grad);
        d_inputs[1].backward(incoming_grad);
    }

    //////////////
    // Multiply
    //////////////
    Tensor Multiply::compute_forward()
    {
        assert(d_inputs.size() == 2 and 
               "Invalid number of inputs, expected 2 but received " + d_inputs.size());

        return d_inputs[0].data() * d_inputs[1].data();
    }

    void Multiply::compute_backward(Tensor const &incoming_grad)
    {
        d_inputs[0].backward(d_inputs[1].data() * incoming_grad);
        d_inputs[1].backward(d_inputs[0].data() * incoming_grad);
    }
}
