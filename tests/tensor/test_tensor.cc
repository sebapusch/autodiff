#include "../test.h"
#include <algorithm>
#include <string>


TEST(Tensor, CostructorInitializesDataVectorCorrectly) {
    Tensor t{{5, 4}, 2.0};
    auto data = t.data();

    EXPECT_EQ(20, data.size());
    EXPECT_THAT(data, ::testing::ContainerEq(vector<double>(20, 2.0)));
}

TEST(Tensor, TensorThrowErrorIndexOutOfBoundsFirstDimension) {
    Tensor t{{5, 4}, 2.0};
    try
    {
        auto a = t[6];
        FAIL();
    }
    catch (runtime_error const &err)
    {
        EXPECT_THAT(err.what(), ::testing::StartsWith("Index out of bounds on dimension 0"));
    }
}

TEST(Tensor, TensorIndex) {
    Tensor t1{{3, 2}, {
        1.0, 2.0,
        5.0, 6.0,
        6.7, 7.5,
    }};

    auto t2 = t1[1];

    EXPECT_THAT(t2.shape(), ::testing::ContainerEq(vector<size_t>{2}));
    EXPECT_THAT(t2.strides(), ::testing::ContainerEq(vector<size_t>{1}));

    double expected[2] = {5.0, 6.0};
    size_t i = 0;
    for_each(t2.begin(), t2.end(), [&expected, &i](double val){
        EXPECT_EQ(expected[i++], val);
    });
}

// TEST(Tensor, TensorOperationDimensionsIncompatible) {
//     Tensor t1{{3, 2, 3}};
//     Tensor t2{{3, 3}};

//     try
//     {
//         prepare_broadcast(t1, t2);
//         FAIL();
//     }
//     catch (runtime_error const &err)
//     {
//         EXPECT_THAT(err.what(), ::testing::StartsWith("Incompatible shapes at axis 1"));
//     }
// }

// TEST(Tensor, TensorOperationDimensionsCompatibleSameShape) {
//     Tensor t1{{3, 2, 3}};
//     Tensor t2{{3, 2, 3}};

//     auto out = prepare_broadcast(t1, t2);

//     auto [shape_a, strides_a] = out.a;
//     auto [shape_b, strides_b] = out.b;
//     auto [shape_res, strides_res] = out.res;

//     EXPECT_THAT(shape_a, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));
//     EXPECT_THAT(shape_b, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));
//     EXPECT_THAT(shape_res, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));

//     EXPECT_THAT(strides_a, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
//     EXPECT_THAT(strides_b, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
//     EXPECT_THAT(strides_res, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
// }

// TEST(Tensor, TensorOperationDimensionsCompatibleBroadcast) {
//     Tensor t1{{3, 2, 3}};
//     Tensor t2{{3, 2, 1}};

//     auto out = calculate_operation_dimensions(t1, t2);

//     auto [shape_a, strides_a] = out.a;
//     auto [shape_b, strides_b] = out.b;
//     auto [shape_res, strides_res] = out.res;

//     EXPECT_THAT(shape_a, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));
//     EXPECT_THAT(shape_b, ::testing::ContainerEq(vector<size_t>{3, 2, 1}));
//     EXPECT_THAT(shape_res, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));

//     EXPECT_THAT(strides_a, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
//     EXPECT_THAT(strides_b, ::testing::ContainerEq(vector<size_t>{2, 1, 0}));
//     EXPECT_THAT(strides_res, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
// }

// TEST(Tensor, TensorOperationDimensionsCompatibleLessDimensions) {
//     Tensor t1{{3, 2, 3}};
//     Tensor t2{   {2, 3}};

//     auto out = calculate_operation_dimensions(t1, t2);

//     auto [shape_a, strides_a] = out.a;
//     auto [shape_b, strides_b] = out.b;
//     auto [shape_res, strides_res] = out.res;

//     EXPECT_THAT(shape_a, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));
//     EXPECT_THAT(shape_b, ::testing::ContainerEq(vector<size_t>{1, 2, 3}));
//     EXPECT_THAT(shape_res, ::testing::ContainerEq(vector<size_t>{3, 2, 3}));

//     EXPECT_THAT(strides_a, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
//     EXPECT_THAT(strides_b, ::testing::ContainerEq(vector<size_t>{0, 3, 1}));
//     EXPECT_THAT(strides_res, ::testing::ContainerEq(vector<size_t>{6, 3, 1}));
// }
