#include "../include/tensorMath.hpp"
#include <iostream>

int main(){
    Tensor t1 = Tensor({2, 2, 2}, 1.0f);
    Tensor t2 = t1.copy(); // Copy
    
    t1.show();
    std::cout << "----" << std::endl;
    t2.show();
    std::cout << "----" << std::endl;

    // t2[{0, 0, 0}] = 9.0f;

    // t2.show();
    // std::cout << "----" << std::endl;
    // t1.show(); // Should still be 1.0f if deep copy worked
    // std::cout << "----" << std::endl;

    t1 = t1+t2;
    std::cout << "t1 = t1 + t2 executed" << std::endl << std::endl;

    t1.show();
    std::cout << "----" << std::endl;
    t2.show();
    std::cout << "----" << std::endl;

    t1 = t1-t2;
    std::cout << "t1 = t1 - t2 executed" << std::endl << std::endl;

    t1.show();
    std::cout << "----" << std::endl;
    t2.show();
    std::cout << "----" << std::endl;

    t1 = t1-t2-t2;
    std::cout << "t1 = t1 - t2 - t2 executed" << std::endl << std::endl;

    t1.show();
    std::cout << "----" << std::endl;
    t2.show();
    std::cout << "----" << std::endl;

    t1 = t1 / 2;
    std::cout << "t1 = t1 / 2 executed" << std::endl << std::endl;

    t1.show();
    std::cout << "----" << std::endl;

    auto t3 = t1.collapse(0);
    std::cout << "t3 = t1.collapse(0) executed" << std::endl << std::endl;

    t3.show();
    std::cout << "----" << std::endl;

    auto t4 = cat(std::vector{t3, t1[0]});
    std::cout << "t4 = cat(std::vector{t3, t1[0]}) executed" << std::endl << std::endl;

    t4.show();
    std::cout << "----" << std::endl;

    auto t5 = stack(std::vector{t4, t4+1}, 2);
    std::cout << "t5 = stack(std::vector{t4, t4+1}, 2) executed" << std::endl << std::endl;

    t5.show();
    std::cout << "----" << std::endl;

    t5.show();
    t5.T().show();


    Tensor ta = Tensor({2, 3}, {0, 1, 2, 10, 11, 12});
    Tensor tb = Tensor({3, 2}, {1, 2, 11, 12, 21, 22});
    ta.show(); tb.show();

    matmul(ta, tb).show();

    ta = Tensor({3}, {0, 1, 2});
    tb = Tensor({3}, {0, 1, 2});
    ta.show(); tb.show();
    
    matmul(ta, tb).show();

    ta = Tensor({2, 3}, {0, 1, 2, 10, 11, 12});
    tb = Tensor({3}, {0, 1, 2});
    ta.show(); tb.show();
    
    matmul(ta, tb).show();

    ta = Tensor({3}, {0, 1, 2});
    tb = Tensor({3, 2}, {0, 1, 10, 11, 20, 21});
    ta.show(); tb.show();
    
    matmul(ta, tb).show();
}