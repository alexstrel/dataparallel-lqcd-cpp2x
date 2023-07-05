#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>
#include <concepts>

//https://en.cppreference.com/w/cpp/container/span/subspan

// Generic container type:
template <typename T>
concept ContainerTp  = requires (T t) {
  t.begin();
  t.end();
  t.data();
  t.size();
};


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

template<ContainerTp T>
struct Foo{
   T data;
   Foo(const int n) : data(n) {}   
   
   Foo(const T& w) : data(w) {} 
    
   void Init() {
     std::iota(data.begin(), data.end(), 0);
   }
   void Print() {
     std::cout << " :::: " << std::endl;	   
     for (auto &i : data) std::cout << i << std::endl;
   }
   
   void PrintPtr() {
     std::cout << " Pointer :: " << data.data() << std::endl; 
   }

   auto Ref() {
      using D = decltype(data[0]);	   
      return Foo<std::span<int>>(std::span{this->data});   
   }

};
 
int main()
{
    char abc[26];
    std::iota(std::begin(abc), std::end(abc), 'A');
    display(abc);
    
    std::cout << " ================= " << std::endl;
    std::vector<int> v = {1,2,3,4,5,6};
    
    //auto s = std::span<int>{v};
    auto ss= foo(v);

    std::cout << v.data() + 4 << std::endl;
    std::cout << ss.data() << " :: " << ss.size() << std::endl;

    for (auto &i : ss) std::cout << i << std::endl;

    auto w = Foo<std::vector<int>> (10);
    w.Init();
    w.Print();
    w.PrintPtr();

    auto ww1 = Foo(w);
    //ww1.Print(); 
    ww1.PrintPtr();

    auto ww2 = Foo(std::span(w.data));
    //ww2.Print();
    ww2.PrintPtr();

    //auto ww2 = Foo<std::span<int>>(std::span(w.data).subspan(w.data.size()/2,2));
    auto ww3 = Foo(std::span(w.data).subspan(w.data.size()/2,2));
    ww3.Print(); 

    auto ww4 = w.Ref();
    ww4.PrintPtr();
}
