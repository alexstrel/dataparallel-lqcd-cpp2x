#include <iostream>
#include <memory>
#include <memory_resource>   // pmr core types
#include <vector>            // pmr::vector
#include <array>
#include <span>

#if 0
std::pmr::monotonic_buffer_resource *getMemoryResource()
{
    static std::array<std::byte, 100000> buffer;
    static std::pmr::monotonic_buffer_resource resource{buffer.data(), buffer.size()};
    return &resource;
}
struct JsonData
{
    std::pmr::vector<int> data{getMemoryResource()};
};
struct XMLData
{
    std::pmr::vector<char> data{getMemoryResource()};
};
#else
//decltype(auto) getMemoryResource()
//{
//    static std::array<std::byte, 100000> buffer;
//    static std::pmr::monotonic_buffer_resource resource{buffer.data(), buffer.size()};
//    return &resource;
//}

//std::pmr::vector<int> data{getMemoryResource()};

//std::pmr::vector<char> data{getMemoryResource()};

#endif

int main() {

    constexpr std::size_t N = 256;

    std::array<std::byte, N> buffer; // a small buffer on the stack
    //
    std::pmr::monotonic_buffer_resource pool{buffer.data(), buffer.size()};
    //
    const std::size_t M = 2;
    std::pmr::vector<int> vec(M, &pool);

    std::cout << buffer.data() << "  " << vec.data() << " " << vec.size() << '\n';

    auto ptr = std::make_shared<char[]>(N);
    //
{
    std::pmr::monotonic_buffer_resource dyn_pool{ptr.get(), N*sizeof(char)};//M<->N ???
    //
    auto pallocF32 = std::pmr::polymorphic_allocator<float>{&dyn_pool};
    //
    //std::pmr::vector<float> vec2(M, &dyn_pool);
    //or 
    std::pmr::vector<float> vec2(64, pallocF32);
    
    printf("%p\n", ptr.get());
    std::cout << "!!" << ptr.get() << "  => " << vec2.data() << " " << vec2.size() << '\n';

    vec2.resize(0);
    dyn_pool.release();

    std::cout << " After " << vec2.data() << " "  << vec2.size() << '\n';
}
{
    const std::size_t offset = 8;
    std::pmr::monotonic_buffer_resource dyn_pool2{ptr.get()+offset, (N/2)*sizeof(char)};

    auto pallocF64 = std::pmr::polymorphic_allocator<double>{&dyn_pool2};

    std::pmr::vector<double> vec3(16, pallocF64);

    printf("%p\n", ptr.get());
    std::cout << "!!!!" << "  => " << vec3.data() << " " << vec3.size() << '\n';

    vec3.resize(0);
    dyn_pool2.release();
}
    //auto s = std::span<double>{vec3};
/*
    const std::size_t offset = 0;

    std::pmr::monotonic_buffer_resource dyn_pool2{ptr.get() + offset, N*sizeof(char)};

    std::pmr::vector<double> vec4(M, &dyn_pool2);

    std::cout << "!!!!" << ptr.get() << "  => " << vec4.data() << " " << vec4.size() << '\n';

*/
    
}

