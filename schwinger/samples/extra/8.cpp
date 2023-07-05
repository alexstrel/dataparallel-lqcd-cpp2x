#include <array>
#include <utility>
#include <iostream>

auto do_something(int x, int y, int z) {
  return (x+y+z);
}

template<std::size_t... Is, typename T, std::size_t N>
auto array_transform_impl(std::index_sequence<Is...>, const std::array<T, N>& arr){
  return do_something(arr[Is]...);
}

int main()
{
  std::array<int, 3> arr = {1, 2, 3};
    
  using Indices = std::make_index_sequence<3>;    
    
  auto res  = array_transform_impl(Indices{}, arr);    
  
  std::cout << " Arr :: " << res << " ! " << std::endl; 
}

