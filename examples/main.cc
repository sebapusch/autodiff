#include "../tensor/tensor.h"
#include "../linalg/linalg.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace autodiff;
using namespace std;

template <typename Engine>
std::vector<double> generate_random_vector(size_t size, double min_val, double max_val, Engine& gen) {
    std::vector<double> random_doubles(size);
    std::uniform_real_distribution<double> dis(min_val, max_val);

    // Use a lambda to capture the distribution and engine
    auto generator = [&]() { return dis(gen); };
    std::generate(random_doubles.begin(), random_doubles.end(), generator);

    return random_doubles;
}

Tensor ReLU(Tensor const &values)
{
    return maximum(values, Tensor{values.shape(), 0});
}

double ReLU_deriv(double val)
{
    return val <= 0 ? 0 : 1;
}

vector<Tensor> epoch(Tensor &W_1, Tensor &W_2, Tensor const &inputs, Tensor const &targets)
{
    Tensor in = concatenate(inputs, Tensor{{1}, 1});

    Tensor a = matmul(W_1, in);
    Tensor z = ReLU(a);


    z = concatenate(z, Tensor{{1}, 1});
    a = concatenate(a, Tensor{{1}, 1});

    Tensor outputs = matmul(W_2, z);

    Tensor err_out = outputs - targets;
    Tensor err_hid = Tensor{z.shape(), 0};

    for (size_t j = 0; j < a.shape()[0]; ++j)
    {
        double err = 0;
        for (size_t k = 0; k < outputs.shape()[0]; ++k)
            err += err_out[k].scalar() * W_2(k, j).scalar();

        err_hid[j] = err * ReLU_deriv(a[j].scalar());
    }

    //cout << err_hid;


    Tensor grad_1 = Tensor{W_1.shape(), 0};
    for (size_t j = 0; j < a.shape()[0] - 1; ++j)
        grad_1[j] = in * err_hid[j];

    // cout << grad_1;

    Tensor grad_out = Tensor{W_2.shape(), 0};
    for (size_t k = 0; k < outputs.shape()[0]; ++k)
        grad_out[k] = z * err_out[k];

    // cout << grad_out;

    return {outputs, grad_1, grad_out};
}

int main()
{
    Tensor X{{5, 2}, {
        -2.50919762,  9.01428613,
         4.63987884,  1.97316968,
        -6.87962719, -6.88010959,
        -8.83832776,  7.32352292,
         2.02230023,  4.16145156,
    }};

    Tensor Y{{5, 3}, {
        -11.72720257,  26.31846104,   1.9252996 ,
          7.99632123,   1.92373599,   8.31626004,
         -5.78214882, -18.60535816, -14.07771627,
        -21.90779115,  23.38371524, -10.30991879,
          2.04125979,  10.26486724,   6.23829595,
    }};

    size_t const D = 2;  // input size
    size_t const M = 8;  // number of hidden units
    size_t const K = 3;  // number of output units

    std::random_device rd;
    std::mt19937 gen(rd());

    Tensor weights_1{{M, D + 1}, {
        4.17022005e-01, 7.20324493e-01, 1.14374817e-04,
        3.02332573e-01, 1.46755891e-01, 9.23385948e-02,
        1.86260211e-01, 3.45560727e-01, 3.96767474e-01,
        5.38816734e-01, 4.19194514e-01, 6.85219500e-01,
        2.04452250e-01, 8.78117436e-01, 2.73875932e-02,
        6.70467510e-01, 4.17304802e-01, 5.58689828e-01,
        1.40386939e-01, 1.98101489e-01, 8.00744569e-01,
        9.68261576e-01, 3.13424178e-01, 6.92322616e-01,
    }};
    Tensor weights_2{{K, M + 1}, {
         0.87638915, 0.89460666, 0.08504421, 0.03905478, 0.16983042, 0.8781425 , 0.09834683, 0.42110763, 0.95788953,
         0.53316528, 0.69187711, 0.31551563, 0.68650093, 0.83462567, 0.01828828, 0.75014431, 0.98886109, 0.74816565,
         0.28044399, 0.78927933, 0.10322601, 0.44789353, 0.9085955 , 0.29361415, 0.28777534, 0.13002857, 0.01936696,
    }};

    const double lr = 0.003;

    string result;

    for (size_t i = 0; i < 100; ++i)
    {
        double loss = 0;
        for (size_t e = 0; e < 5; ++e)
        {
            vector<Tensor> res = epoch(weights_1, weights_2, X[e], Y[e]);
            Tensor &out = res[0];
            Tensor &grad_1 = res[1];
            Tensor &grad_2 = res[2];

            Tensor diff = (out - Y[e]).power(2);

            loss += diff.sum();

            weights_1 -= grad_1 * lr;
            weights_2 -= grad_2 * lr;
        }

        result += "loss epoch " + to_string(i) + ": " + to_string(loss) + "\n";
    }
    cout << result << weights_1;
}


// int main()
// {
//     Tensor t{{2, 3}, 6};

//     cout << t;

//     t[1] = Tensor{{3}, 2};

//     cout << t;

//     t[1][2] = 17;

//     cout << t;

//     double &val = t[0][1].scalar();
//     val = 89;

//     cout << t[0][1].scalar() << "\n";
// }
