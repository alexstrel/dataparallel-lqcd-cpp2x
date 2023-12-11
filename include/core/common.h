#pragma once

#include <concepts> 
#include <type_traits>
#include <span>

#include <ranges>
#include <vector>
#include <iostream>
#include <complex>
#include <algorithm>
#include <execution>

#include <random>

#include <experimental/mdspan>

#include <limits>

constexpr int min_int                = std::numeric_limits<int>::min();
constexpr std::size_t min_sizet      = std::numeric_limits<std::size_t>::min();

//some defs:
constexpr std::size_t invalid_spin   = min_sizet;
constexpr std::size_t invalid_color  = min_sizet;
constexpr std::size_t invalid_dir    = min_sizet;
constexpr std::size_t invalid_parity = min_sizet;

namespace stdex = std::experimental;

// Extended floaitng point type:
template <typename T>
concept FloatTp = std::is_floating_point_v<T>;

// Generic complex type:
template <typename T>
concept ComplexTp = requires {
    typename T::value_type;
    
    requires FloatTp<typename T::value_type>;
    
    { std::declval<T>().real() } -> std::convertible_to<typename T::value_type>;
    { std::declval<T>().imag() } -> std::convertible_to<typename T::value_type>;
};

// Generic arithmetic type:
template <typename T>
concept ArithmeticTp = FloatTp<T> or ComplexTp<T>;

// Generic (complex valued) container type (inlcudes both allacotar aware containers and views)
template <typename T>
concept GenericContainerTp = requires{
    typename T::value_type;
    typename T::size_type;
    typename T::iterator;
    
    //requires ComplexTp<typename T::value_type>;//TODO : this does not work with block fields

    { std::declval<T>().data()  }  -> std::same_as<typename T::value_type*>;
    { std::declval<T>().size()  }  -> std::same_as<typename T::size_type>;
    //
    { std::declval<T>().begin() }  -> std::convertible_to<typename T::iterator>;
    { std::declval<T>().end()   }  -> std::convertible_to<typename T::iterator>;
}; 

template<typename T>
concept is_allocator_aware_type = requires{
    typename T::allocator_type;
    //requires std::same_as<decltype(std::declval<T>().get_allocator()), typename T::allocator_type>; //WHY this fails?..
} 
and not requires { 
  requires std::same_as<typename T::allocator_type, std::pmr::polymorphic_allocator<typename T::value_type>>;
  //{std::declval<typename T::allocator_type>().template new_object<typename T::value>()}; 
};

//Allocator aware container type:
template <typename T>
concept ContainerTp = GenericContainerTp<T> and is_allocator_aware_type<T>;

//Polymorphic allocator aware container type:

template <typename T>
concept is_pmr_allocator_aware_type = requires {
    typename T::allocator_type;
    requires std::same_as<typename T::allocator_type, std::pmr::polymorphic_allocator<typename T::value_type>>;
    //requires std::same_as<decltype(std::declval<T>().get_allocator()), typename T::allocator_type>; //WHY this fails?..    
};


template <typename T>
concept PMRContainerTp = GenericContainerTp<T> and is_pmr_allocator_aware_type<T>;

//
template <typename T>
concept is_memory_non_owning_type = not (is_allocator_aware_type<T> or is_pmr_allocator_aware_type<T>);

//Non-owning container type:
template <typename T>
concept ContainerViewTp = GenericContainerTp<T> and is_memory_non_owning_type<T>;

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class is_mdspan : public std::false_type {};

template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
class is_mdspan<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>> : public std::true_type {};

// MDspan concept:
template <typename T>
concept MDViewTp = is_mdspan<std::remove_cvref_t<T>>::value;

