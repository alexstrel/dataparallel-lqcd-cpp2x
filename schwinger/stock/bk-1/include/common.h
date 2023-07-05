#pragma once

#include <concepts> 
#include <type_traits>
#include <span>

#include <ranges>
#include <vector>
#include <iostream>
#include <complex>
#include <algorithm>

#include <random>

#include <experimental/mdspan>

namespace stdex = std::experimental;

// Simple arithmetic type
template <typename T>
concept FloatTp = requires{
  requires std::is_floating_point_v<T>;
};

// Simple complex type
template <typename T>
concept ComplexTp    = requires (T t) {
  requires FloatTp<decltype(t.real())>;
  requires FloatTp<decltype(t.imag())>;  
};

// Simple genetic arithmetic type
template <typename T>
concept ArithmeticTp = FloatTp<T> || ComplexTp<T>;


// Generic container type:
template <typename T>
concept GenericContainerTp  = requires (T t) {
  t.begin();
  t.end();
  t.data();
  t.size();
};


