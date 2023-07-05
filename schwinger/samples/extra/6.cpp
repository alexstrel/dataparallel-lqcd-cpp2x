#include <array>
#include <tuple>
#include <utility>

template<std::size_t... Is, typename Array>
auto array_to_tuple_impl(std::index_sequence<Is...>, const Array& arr)
{
    return std::make_tuple(arr[Is]...);
}

template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
auto array_to_tuple(const std::array<T, N>& arr)
{
    return array_to_tuple_impl(Indices{}, arr);
}

int main()
{
    std::array<int, 3> arr = {1, 2, 3};
    auto tuple = array_to_tuple(arr);
}

