#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

//https://en.cppreference.com/w/cpp/container/span/subspan

void display(std::span<const char> abc)
{
    const auto columns{ 20U };
    const auto rows{ abc.size() - columns + 1 };

    for (auto offset{ 0U }; offset < rows; ++offset) {
        std::ranges::for_each(
            abc.subspan(offset, columns),
            std::putchar
        );
        std::putchar('\n');
    }
}

auto foo(std::vector<int> &v){
  return std::span{v}.subspan(v.size() / 2, v.size() / 2);
}
 
int main()
{
    char abc[26];
    std::iota(std::begin(abc), std::end(abc), 'A');
    display(abc);
    
    std::cout << " ================= " << std::endl;
    std::vector<int> v = {1,2,3,4,5,6,7,8,9};
    
    //auto s = std::span<int>{v};
    auto ss= foo(v);

    std::cout << v.data() + 4 << std::endl;
    std::cout << ss.data() << " :: " << ss.size() << std::endl;

    for (auto &i : ss) std::cout << i << std::endl;
}
