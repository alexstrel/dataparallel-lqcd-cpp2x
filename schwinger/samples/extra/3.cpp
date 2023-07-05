#include <iostream>
#include <memory_resource>
#include <vector>

static auto upstream = std::pmr::get_default_resource();


constexpr std::size_t nbytes = 4096;

int main() {
    std::pmr::monotonic_buffer_resource resource_1{upstream};

    // Use the memory resource with a polymorphic allocator
    std::pmr::vector<int> vec_1{16, &resource_1};

    int x = 0;
    for (auto &element : vec_1) {
      element = x++;
    }

    printf("%p %p\n", vec_1.data(), vec_1.end());
    for (int element : vec_1) {
        std::cout << element << ' ';
    }
    std::cout << '\n';

    std::pmr::vector<int> vec_2{16, &resource_1};

    x = 10;
    for (auto &element : vec_2) {
      element = x++;
    }

    printf("%p %p\n", vec_2.data(), vec_2.end());
    for (int element : vec_2) {
        std::cout << element << ' ';
    }
    std::cout << '\n';

    std::pmr::monotonic_buffer_resource resource_2{upstream};  

    std::pmr::vector<int> vec_3{16, &resource_2};

    x = 30;
    for (auto &element : vec_3) {
      element = x++;
    }

    printf("%p %p\n", vec_3.data(), vec_3.end());
    for (int element : vec_3) {
        std::cout << element << ' ';
    }
    std::cout << '\n';


    // Reset the memory resource
    resource_1.release();

    std::pmr::vector<int> vec_4{16, &resource_1};

    x = 50;
    for (auto &element : vec_4) {
      element = x++;
    }

    printf("%p %p\n", vec_4.data(), vec_4.end());
    for (int element : vec_4) {
        std::cout << element << ' ';
    }
    std::cout << '\n';

    printf("%p %p\n", vec_1.data(), vec_1.end());
    for (int element : vec_1) {
        std::cout << element << ' ';
    }
    std::cout << '\n';

    return 0;
}


