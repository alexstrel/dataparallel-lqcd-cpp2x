#include <iostream>
#include <memory>
#include <memory_resource>   // pmr core types
#include <vector>            // pmr::vector
#include <array>
#include <span>
#include <algorithm>

#include <complex>

constexpr std::size_t buffer_size = 256;
//
static std::shared_ptr<std::byte[]> ptr = nullptr;

int main() {

    std::array<std::byte, buffer_size> buffer; // a small buffer on the stack
    //
    std::pmr::monotonic_buffer_resource pool{buffer.data(), buffer.size()};
    //
    std::pmr::vector<int> vec(2, &pool);

    std::cout << buffer.data() << "  " << vec.data() << " " << vec.size() << '\n';

    ptr = std::make_shared<std::byte[]>(buffer_size);
    //
    std::pmr::monotonic_buffer_resource dyn_pool{ptr.get(), buffer_size*sizeof(std::byte)};//M<->N ???
    //
    constexpr std::size_t vec2_size =  buffer_size / sizeof(float);
    //
    //auto pmr_allocF32 = std::pmr::polymorphic_allocator<float>{&dyn_pool};
    //std::pmr::vector<float> vec2(vec2_size, pmr_allocF32);
    // or
    std::pmr::vector<float> vec2(vec2_size, &dyn_pool);

    std::for_each(vec2.begin(), vec2.end(), [x=0.0f] (auto &i) mutable { i = x; x += 1.1f; }  );
    
    printf("%p\n", ptr.get());
    std::cout << "!!" << ptr.get() << "  => " << vec2.data() << " " << vec2.size() << '\n';

    for(auto& x : vec2) { std::cout << x << std::endl; }
    //vec2.resize(0);
    //dyn_pool.release();
    //std::cout << " After " << vec2.data() << " "  << vec2.size() << '\n';

    const std::size_t corrected_buffer_size = buffer_size / 2;

    const std::size_t vec3_size             = corrected_buffer_size / sizeof(double); 

    const std::size_t offset                = corrected_buffer_size;
    
    std::pmr::monotonic_buffer_resource dyn_pool_with_offset{ptr.get()+offset, corrected_buffer_size*sizeof(std::byte)};

    auto pmr_allocF64 = std::pmr::polymorphic_allocator<double>{&dyn_pool_with_offset};

    std::pmr::vector<double> vec3(vec3_size, pmr_allocF64);

    printf("%p\n", ptr.get());
    std::cout << "!!!!" << "  => " << vec3.data() << " " << vec3.size() << '\n';

    std::for_each(vec3.begin(), vec3.end(), [x=100.0f] (auto &i) mutable { i = x; x += 1.1f; }  );

    for(auto& x : vec3) { std::cout << x << std::endl; }

    vec3.resize(0);
    //dyn_pool2.release();
    //
    auto s = std::span<float>{vec2};
    for(auto& x : s) { std::cout << x << std::endl; }

    std::pmr::vector<float> vec4(vec2_size, &dyn_pool);

    std::pmr::vector<float> vec5(vec3_size, &dyn_pool_with_offset);

    std::vector<float> vec6(vec2_size);

    using pmr_vector = std::pmr::vector<float>; 
 
    auto s2 = std::span<float>{vec6};
    
    std::cout << std::boolalpha;    

    std::cout << std::is_same<decltype(vec2), decltype(vec4)>::value << std::endl; 
    //
    std::cout << std::is_same<decltype(vec2), decltype(vec5)>::value << std::endl;
    //
    std::cout << std::is_same<decltype(vec2), decltype(vec6)>::value << std::endl;
    //
    std::cout << std::is_same<decltype(s), decltype(s2)>::value << std::endl;
    //
    std::cout << std::is_same<decltype(vec2), pmr_vector>::value << std::endl;

    std::pmr::vector<std::complex<float>> vec10(4,  &dyn_pool);
    for(auto& x : vec10) { std::cout << x << std::endl; }

    std::pmr::vector<float> vec7{vec2};

    for(auto& x : vec7) { std::cout << x << std::endl; }

    std::vector<float> vec8(2, std::allocator<float>());
    for(auto& x : vec8) { std::cout << x << std::endl; }

    using allc_type = decltype(vec2.get_allocator());

    std::cout << std::is_same<allc_type, std::pmr::polymorphic_allocator<typename  pmr_vector::value_type> >::value << std::endl;

    std::cout << std::is_same<typename  pmr_vector::allocator_type, std::pmr::polymorphic_allocator<typename  pmr_vector::value_type> >::value << std::endl;

    //
    std::vector<std::complex<float>> vec11(8);
    //
    std::cout << "A:: " << vec11.size() << std::endl;

    std::pmr::vector<float> vec12(&dyn_pool);

    vec11.resize(0);
    std::cout << "B:: " << vec11.size() << std::endl;

}

