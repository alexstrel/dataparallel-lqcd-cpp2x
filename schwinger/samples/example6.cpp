#include <cstdint>
#include <iterator>
#include <vector>
#include <iostream>
#include <memory>

template <typename Tp>
class Allocator
{
  private:
    Tp* ptr;
    std::size_t size;

  public:
    using pointer    = Tp*;
    using value_type = Tp;

    Allocator(Tp* ptr, std::size_t size) : ptr(ptr), size(size) {}

    Allocator(const Allocator& other) noexcept : ptr(other.ptr), size(other.size) {};

    template<typename Tp1>
    Allocator(const Allocator<Tp1>& other) noexcept : ptr(other.ptr), size(other.size) {};

    template<typename Tp1>
    Allocator& operator     = (const Allocator<Tp1>& other) { return *this; }
    Allocator<Tp>& operator = (const Allocator& other) { return *this; }
    ~Allocator() {}

    pointer allocate(std::size_t n, const void* hint = 0) {return ptr;}
    void deallocate(Tp* ptr, std::size_t n) {}

    std::size_t max_size() const {return size;}
};


int main()
{
  //
  std::allocator<char> char_alloc;
  auto ptr = std::allocate_shared<char>(char_alloc, static_cast<std::size_t>(128));
  //
  std::vector<int, Allocator<int>> vec1(0, Allocator<int>(reinterpret_cast<int*>(ptr.get()), 4));
  vec1.push_back(1);
  std::cout<<"Vec[0]: "<< vec1[0]<<"\n";
  std::cout<<"Ptr: " << vec1.data() <<"\n";
   
  auto allocf =  Allocator<float>(reinterpret_cast<float*>(ptr.get()), 4);
  std::vector<float, decltype(allocf)> vec2(0, allocf);
  vec2.push_back(1.1f);
  std::cout<<"Vec[0]: "<< vec2[0]<<"\n";
  std::cout<<"Ptr: " << vec2.data() <<"\n";

  return 0;
}

