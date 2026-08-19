#include <chrono>
#include <iostream>
#include "Arrays/ndarrays.hpp"

int main(){
    Array<float> A({1000, 10000}, 1);
    Array<float> B({10000, 1000}, 1);
    Array<float> C({1, 1000000}, 1);
    Array<float> D({1, 1000000}, 1);
    Array<float> E({1000000}, 1);
    Array<float> F({1000000}, 1);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto lmntw_t1 = high_resolution_clock::now();
    Array<float> lmntw_result = C * D / E + F - C;
    auto lmntw_t2 = high_resolution_clock::now();

    // Getting number of milliseconds as a double.
    duration<double, std::milli> lmntw_exe_time = lmntw_t2 - lmntw_t1;

    std::cout << "Element-wise operations time: " << lmntw_exe_time.count() << "ms\n";

    auto mmult_t1 = high_resolution_clock::now();
    Array<float> mmult_result = nd::matmul(A, B);
    auto mmult_t2 = high_resolution_clock::now();
    duration<double, std::milli> mmult_exe_time = mmult_t2 - mmult_t1;

    std::cout << "Matrix multiplication time: " << mmult_exe_time.count() << "ms\n";

    std::cout << "Tests" << std::endl;
    Array<float> a({2, 3}, {1, 2, 3, 4, 0, 6});
    a.show();
    (a == Array<float>({3}, {1, 0, 3})).show();

    nd::max(a).show();
    nd::min(a).show();
    nd::sum(a).show();

    nd::max(a, {0}).show();
    nd::min(a, {0}).show();
    nd::sum(a, {0}).show();

    nd::max(a, {1}).show();
    nd::min(a, {1}).show();
    nd::sum(a, {1}).show();

    nd::max(a, {0}, true).show();
    nd::min(a, {0}, true).show();
    nd::sum(a, {0}, true).show();

    nd::max(a, {1}, true).show();
    nd::min(a, {1}, true).show();
    nd::sum(a, {1}, true).show();

    (a == nd::max(a, {0})).show();
    (a == nd::max(a, {0}, true)).show();
    (a == nd::max(a, {1}, true)).show();

    Array<bool> b({2, 2, 2}, {1, 0, 1, 0, 1, 0, 1, 0});
    b.show();
}
