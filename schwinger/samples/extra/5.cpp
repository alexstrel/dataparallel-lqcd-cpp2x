#include <iostream>
#include <memory>
#include <memory_resource>
#include <vector>

constexpr std::size_t alignment_request = std::alignment_of_v<long double>;//or alignof(std::max_align_t)


class UpstreamMemoryResource : public std::pmr::memory_resource {
  protected:
    //
    std::size_t nBytes = 0;
    
    void* do_allocate(std::size_t nbytes, std::size_t alignment = alignment_request) override {//alignof(std::max_align)
      nBytes = ( (nbytes + alignment - 1) / alignment ) *  alignment;
      //
      printf("Allocate %d bytes.. (requested %d)\n", nBytes, nbytes);
      return std::aligned_alloc(alignment, nBytes);
    }
    
    void do_deallocate(void* ptr, std::size_t nbytes, std::size_t alignment = alignment_request) override {
      //
      printf("Free %d bytes..\n", nBytes);      
      return std::free(ptr);
    }
    
    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override { return (this == &other); }
    
  public:
  
    UpstreamMemoryResource()                                = default;
    UpstreamMemoryResource(const UpstreamMemoryResource &)  = default;
    UpstreamMemoryResource(UpstreamMemoryResource &&)       = default;     
};

//static UpstreamMemoryResource upstream;
static auto upstream = UpstreamMemoryResource{};

constexpr std::size_t nbytes = 4096;

int main() {
    std::cout << alignof(std::max_align_t) << " :: " << alignment_request << '\n'; 
    
    //UpstreamMemoryResource upstream;
    //
    std::pmr::monotonic_buffer_resource resource_1(&upstream);

    // Use the memory resource with a polymorphic allocator
    std::pmr::vector<int> vec_1(531, &resource_1);

    int x = 0;
    for (auto &element : vec_1) {
      element = x++;
    }

    printf("vec1 %p %p\n", vec_1.data(), vec_1.end());
    for (int element : vec_1) {
        std::cout << element << ' ';
    }
    std::cout << '\n';

    std::pmr::vector<int> vec_2{130, &resource_1};

    x = 10;
    for (auto &element : vec_2) {
      element = x++;
    }

    printf("vec2 %p %p\n", vec_2.data(), vec_2.end());
    for (int element : vec_2) {
        std::cout << element << ' ';
    }
    std::cout << '\n';

    std::pmr::monotonic_buffer_resource resource_2(&upstream);  

    std::pmr::vector<int> vec_3{16, &resource_2};

    x = 30;
    for (auto &element : vec_3) {
      element = x++;
    }

    printf("vec3 (new resource) %p %p\n", vec_3.data(), vec_3.end());
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

    printf("(vec4 after release) %p %p\n", vec_4.data(), vec_4.end());
    for (int element : vec_4) {
        std::cout << element << ' ';
    }
    std::cout << '\n';
    
/*
    printf("%p %p\n", vec_1.data(), vec_1.end());
    for (int element : vec_1) {
        std::cout << element << ' ';
    }
    std::cout << '\n';
*/
    return 0;
}


