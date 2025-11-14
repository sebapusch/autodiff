#include "variable/variable.h"
#include "tensor/tensor.h"
#include <iostream>
#include <print>

using namespace autodiff;
using namespace std;

int main()
{
    auto a = Variable(Tensor({1},{5}));
    auto b = Variable(Tensor({1},{7}));

    auto c = a + b;

    cout << c.data();

//    println("{}", c.data()); @todo make this shit work
}
