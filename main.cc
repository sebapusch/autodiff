#include "variable/variable.h"
#include "tensor/tensor.h"
#include <iostream>
#include <print>

using namespace autodiff;
using namespace std;

int main()
{
    Variable a(8);
    Variable b(19);

    auto c = a * b;

    cout << "a = " << a.data().scalar() 
         << "\nb = " << b.data().scalar() 
         << "\nc = a * b = " << c.data().scalar() << "\n";

    c.backward();

    if (auto grad = a.grad())
        cout << "grad a = " << grad.value().scalar() << "\n";
    else
        cout << "no grad\n";
}
