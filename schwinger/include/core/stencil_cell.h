#pragma once

#include <common.h>

namespace impl 
{

template<int... M>
consteval int get_cell_size() {

  std::array<int, sizeof... (M)> r{M...};
  int s = 1;

  for (int &i : r)  s = s*i;

  return s;
}	

template <ComplexTp T, int... M>
class StencilCell {
   public :
     using value_type = T;
     //local dimensions
     static constexpr int D{sizeof...(M)};
     static constexpr std::array<int, D> m{M...};
     static constexpr int cell_size{get_cell_size<M...>()};
     //local (right) views:
     using MDView   =  stdex::mdspan<T,       stdex::extents<indx_type, M...>, stdex::layout_right, stdex::default_accessor<T>>;
     using MDCView  =  stdex::mdspan<const T, stdex::extents<indx_type, M...>, stdex::layout_right, stdex::default_accessor<const T>>;

     std::array<T, cell_size> v;

     StencilCell()                             = default;
     StencilCell(const StencilCell<T, M...> &) = default;
     StencilCell(StencilCell<T, M...> &&)      = default;
   
     //basic accessors
     constexpr T &operator[](const int i) { return v[i]; }
     constexpr const T &operator[](const int i) const { return v[i]; }

     constexpr int size() const { return getsize<M...>(); }   

     auto operator=(const StencilCell&) -> StencilCell& = default;
     auto operator=(StencilCell&&     ) -> StencilCell& = default;
     
     template<bool is_constant = false>
     decltype(auto) accessor() const {
       if constexpr (is_constant) {
         return MDCView(v.data()); 
       } else {
         return MDView(const_cast<T*>(v.data()));        
       }
     } 
#if 1     
     // naive load method:
     inline decltype(auto) load() const {
       if constexpr (cell_size == 1) {
         exit(-1);
       }
       return v;
     }    

     // naive store method:
     inline void store( const std::array<T, cell_size> &u) {
#pragma unroll     
       for(int i = 0; i < cell_size; i++) v[i] = u[i];
     }    
#else
     // naive load method:
     decltype(auto) load() const {
       return v[0];
     }    

     // naive store method:
     void store( const T &u) {
       v[0] = u;
     } 
#endif
   
};

}//end namespace impl
