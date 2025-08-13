#include "tensor/tensor.h"
#include "linalg/linalg.h"
#include <iostream>

using namespace autodiff;
using namespace std;

int main()
{
    Tensor a{{1, 2, 2}, {
        0, 1,
        2, 3
    }};
    Tensor b{{1, 1, 2}, {
        4, 5
    }};

    cout << a;
    cout << b;
    cout << concatenate(a, b, nullopt);
    cout << "\n";

    a = Tensor{{4, 2, 3}, {
        0, 1, 2,
        3, 4, 5,

        6, 7, 8,
        9, 10, 11,

        12, 13, 14,
        15, 16, 17,

        18, 19, 20,
        21, 22, 23
    }};

    b = {{1, 2, 3}, {
        24, 25, 26,
        27, 28, 29
    }};
    cout << concatenate(a, b);

    Tensor c = {{3, 5}, {
        0,  1,  2,  3,  4,
        5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
    }};
    cout << matmul(a, c);
    cout << a + b;
    cout << "\n";
}
