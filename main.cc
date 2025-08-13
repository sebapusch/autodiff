#include "tensor/tensor.h"
#include "linalg/linalg.h"
#include <iostream>

using namespace autodiff;
using namespace std;

int main()
{
    Tensor a{{2, 3}, {
        2, 3, 5,
        2, 5, 7,
    }};

    Tensor b{{2, 3, 4}, {
        3, 1, 7, 5,
        4, 4, 1, 1,
        9, 1, 8, 8,

        4, 5, 3, 2,
        2, 4, 5, 6,
        3, 4, 6, 2,
    }};

    auto res = matmul(a, b);

    cout << res;
}
