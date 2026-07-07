#include <chrono>
#include <iostream>
#include "Arrays/cpp_arrays.hpp"

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
    Array<float> mmult_result = matmul(A, B);
    auto mmult_t2 = high_resolution_clock::now();
    duration<double, std::milli> mmult_exe_time = mmult_t2 - mmult_t1;

    std::cout << "Matrix multiplication time: " << mmult_exe_time.count() << "ms\n";

    std::cout << "Tests" << std::endl;
    Array<float> a({2, 3}, {1, 2, 3, 4, 0, 6});
    (a == Array<float>({3}, {1, 0, 3})).show();
    max(a).show();
    min(a).show();
    sum(a).show();
}

// int main(){
//     Array<float> t1 = Array<float>({2, 2, 2}, 1.0f);
//     Array<float> t2 (t1);
    
//     t1.show();
//     std::cout << "----" << std::endl;
//     t2.show();
//     std::cout << "----" << std::endl;

//     // t2[{0, 0, 0}] = 9.0f;

//     // t2.show();
//     // std::cout << "----" << std::endl;
//     // t1.show(); // Should still be 1.0f if deep copy worked
//     // std::cout << "----" << std::endl;

//     t1 = t1+t2;
//     std::cout << "t1 = t1 + t2 executed" << std::endl << std::endl;

//     t1.show();
//     std::cout << "----" << std::endl;
//     t2.show();
//     std::cout << "----" << std::endl;

//     t1 -= t2;
//     std::cout << "t1 -= t2 executed" << std::endl << std::endl;

//     t1.show();
//     std::cout << "----" << std::endl;
//     t2.show();
//     std::cout << "----" << std::endl;

//     t1 = t1-t2-t2;
//     std::cout << "t1 = t1 - t2 - t2 executed" << std::endl << std::endl;

//     t1.show();
//     std::cout << "----" << std::endl;
//     t2.show();
//     std::cout << "----" << std::endl;

//     t1 = t1 / 2;
//     std::cout << "t1 = t1 / 2 executed" << std::endl << std::endl;

//     t1.show();
//     std::cout << "----" << std::endl;


//     Array<float> ta = Array<float>({2, 3}, {0, 1, 2, 10, 11, 12});
//     Array<float> tb = Array<float>({3, 2}, {1, 2, 11, 12, 21, 22});
//     ta.show(); tb.show();

//     matmul(ta, tb).show();

//     ta = Array<float>({3}, {0, 1, 2});
//     tb = Array<float>({3}, {0, 1, 2});
//     ta.show(); tb.show();
    
//     matmul(ta, tb).show();

//     ta = Array<float>({2, 3}, {0, 1, 2, 10, 11, 12});
//     tb = Array<float>({3}, {0, 1, 2});
//     ta.show(); tb.show();
    
//     matmul(ta, tb).show();

//     ta = Array<float>({3}, {0, 1, 2});
//     tb = Array<float>({3, 2}, {0, 1, 10, 11, 20, 21});
//     ta.show(); tb.show();
    
//     matmul(ta, tb).show();
// }