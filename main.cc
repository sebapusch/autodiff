#include "tensor/tensor.h"
#include "linalg/linalg.h"
#include <iostream>

using namespace autodiff;
using namespace std;

int main()
{
    Tensor a{{3, 2, 3}, {
        0, 1, 2,
        3, 4, 5,

        6, 7, 8,
        9, 10, 11,

        12, 13, 14,
        15, 16, 17
    }};

    cout << a;

    cout << "\n";

    Tensor b = a[2];

    Tensor c = b[1];

    cout << b << "\n" << c << "\n";
}
