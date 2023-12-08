#pragma once

#include <core/cartesian_product.hpp>

namespace impl{

  /** Convert tuple into std::array in the inverse order:
  */
  template<std::size_t... Id>
  inline decltype(auto) get_cartesian_coords(std::index_sequence<Id...>, const auto& x) {
    return std::array<int, sizeof... (Id)>{{std::get<sizeof...(Id)-Id-1>(x)... }};
  }

  template<std::size_t nDim>
  inline decltype(auto) convert_coords(const auto &x) {
    return get_cartesian_coords(std::make_index_sequence<nDim>{}, x);
  } 
  
  inline decltype(auto) cartesian_2d_view(const std::array<int, 2> &x){
      
    auto X = std::views::iota(0, x[0]);
    auto Y = std::views::iota(0, x[1]);

    return std::views::cartesian_product(Y, X);//Y is the slowest index, X is the fastest  
  }
  
  inline decltype(auto) cartesian_4d_view(const std::array<int, 4> &x){
      
    auto X = std::views::iota(0, x[0]);
    auto Y = std::views::iota(0, x[1]);
    auto Z = std::views::iota(0, x[2]);
    auto T = std::views::iota(0, x[3]);      

    return std::views::cartesian_product(T, Z, Y, X);//T is the slowest index, X is the fastest  
  }  
  
  inline decltype(auto) restricted_cartesian_4d_view(const std::array<int, 4> &begin_x, const std::array<int, 4> &end_x){
      
    auto X = std::views::iota(begin_x[0], end_x[0]);//excluded end_x
    auto Y = std::views::iota(begin_x[1], end_x[1]);
    auto Z = std::views::iota(begin_x[2], end_x[2]);
    auto T = std::views::iota(begin_x[3], end_x[3]);      

    return std::views::cartesian_product(T, Z, Y, X);//T is the slowest index, X is the fastest  
  }    
    
   
} // 
