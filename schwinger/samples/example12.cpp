#include <iostream>
#include <memory>
#include <memory_resource>   // pmr core types
#include <vector>            // pmr::vector
#include <array>
#include <span>
#include <algorithm>

#include <complex>

constexpr std::size_t buffer_size = 8192*16;
//
static std::shared_ptr<std::byte[]> ptr = nullptr;

int main() {

  ptr = std::make_shared<std::byte[]>(buffer_size);
  //
  std::pmr::monotonic_buffer_resource dyn_pool{ptr.get(), buffer_size*sizeof(std::byte)};
  //
  constexpr std::size_t vec_size =  buffer_size / sizeof(float);
  //
  // or
  std::pmr::vector<float> vec(vec_size, &dyn_pool);

  std::for_each(vec.begin(), vec.end(), [x=0.0f] (auto &i) mutable { i = x; x += 1.1f; }  );
    
  printf("%p\n", ptr.get());
  std::cout << "!!" << ptr.get() << "  => " << vec.data() << " " << vec.size() << '\n';

  for(auto& x : vec) { if ( x < 64.f ) std::cout << x << std::endl; }
  //
  std::pmr::monotonic_buffer_resource next_dyn_pool{ptr.get(), (buffer_size)*sizeof(std::byte)}; 
  
  std::pmr::vector<float> next_vec(vec_size, &next_dyn_pool);

  std::for_each(next_vec.begin(), next_vec.end(), [x=0.0f] (auto &i) mutable { i = x; x -= 1.1f; }  );
    
  printf("%p\n", ptr.get());
  std::cout << "!!" << ptr.get() << "  => " << next_vec.data() << " " << next_vec.size() << '\n'; 
  
  for(auto& x : next_vec) { if ( x > -64.f ) std::cout << x << std::endl; }    
  
}

