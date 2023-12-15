#pragma once

#include <ranges>
//
#include <core/color_spinor.h>

template<typename T> concept Indx = std::is_same_v<T, int> or std::is_same_v<T, std::size_t>;

template<typename T> concept RangesTp = std::ranges::contiguous_range<T> and std::ranges::sized_range<T> and std::ranges::viewable_range<T>;

template<GenericFieldTp F, bool is_constant = false, int bSize = 1>
class FieldAccessor {
  public:
    using data_tp  = typename F::data_tp;

    using SpinorTp = impl::ColorSpinor<data_tp, F::Ncolor(), F::Nspin(), bSize>;
    using LinkTp   = impl::ColorMatrix<data_tp, F::Ncolor(), bSize>;
    
    using Indices  = std::make_index_sequence<F::Ndim()>;      

    using AccessorTp      = typename std::remove_cvref_t< decltype( std::declval<F>().template Accessor<is_constant>()) >; 
  
    AccessorTp       field_accessor;
    //
    
    FieldAccessor(const F &field ) : field_accessor(field.template Accessor<is_constant>()) {}        

    inline constexpr decltype(auto) Extent(int d) const { return field_accessor.extent(d); }              
      
    /**
       @brief 2-d accessor functor
       @param[in] x coords
       @param[in] y coords
       @param[in] i spin dof or dir      
       @return Complex number at this spin and color index
    */
    template <Indx ...indx_tp>
    inline data_tp& operator()(indx_tp ...i) { return field_accessor(i...); }

    /**
       same as above but constant 
    */
    template <Indx ...indx_tp>
    inline data_tp& operator()(indx_tp ...i) const { return field_accessor(i...); }    



    template<std::size_t... Idxs, FieldType type = F::Type()>
    requires (type == FieldType::StaggeredSpinorFieldType)
    inline decltype(auto) load_parity_spinor(std::index_sequence<Idxs...>, const RangesTp auto& x) const {  
    
      using idx_type = decltype(x[0]);    
    
      constexpr int ncolor = F::Ncolor();

      auto spinor = [this, x=x]() {
        std::array<data_tp, ncolor*bSize> tmp;
#pragma unroll
        for (int i = 0; i < ncolor; i++) {
#pragma unroll
          for (int b = 0; b < bSize; b++ ) {//bSize = 1
            tmp[i*bSize+b] = this->field_accessor(x[Idxs]..., static_cast<idx_type>(i));
          }
        } return tmp; } ();    

      return SpinorTp(spinor);
    }
    
    template<FieldType field_type = F::type()>
    requires (field_type == FieldType::StaggeredSpinorFieldType)        
    inline decltype(auto) operator()(const RangesTp auto &x) const {
      return load_parity_spinor(Indices{}, x);    
    }

    template<std::size_t... Idxs, FieldType field_type = F::Type()>
    requires (field_type == FieldType::VectorFieldType)
    inline decltype(auto) load_parity_link(std::index_sequence<Idxs...>, const RangesTp auto& x, const int &d, const int &parity) const {

      using idx_type = decltype(x[0]); 
      
      constexpr int ncolor = F::Ncolor();

      auto link = [this, x=x, d=d, p=parity]() {
        std::array<data_tp, ncolor*ncolor*bSize> tmp;
#pragma unroll
        for (int j = 0; j < ncolor; j++) {
#pragma unroll
          for (int i = 0; i < ncolor; i++) {
#pragma unroll
            for (int b = 0; b < bSize; b++ ) {//bSize = 1
              tmp[(j*ncolor+i)*bSize+b] = this->field_accessor(x[Idxs]...,  static_cast<idx_type>(j),  static_cast<idx_type>(i),  const_cast<idx_type>(d),  const_cast<idx_type>(p));
            }
          }
        } return tmp; } ();
        
      return LinkTp(link);
    }
    
    template<FieldType field_type = F::type()>
    requires (field_type == FieldType::VectorFieldType)
    inline decltype(auto) operator()(const RangesTp auto &x, const int &d, const int &p) const {
      return load_parity_link(Indices{}, x, d, p);
    }

    template<std::size_t... Idxs, FieldType field_type = F::type()>
    requires (field_type == FieldType::StaggeredSpinorFieldType)
    inline data_tp& store_staggered_spinor_component(std::index_sequence<Idxs...>, const RangesTp auto& x, const int c) {
      return this->field_accessor(x[Idxs]..., c);
    }

    template<FieldType field_type = F::type()>
    requires (field_type == FieldType::StaggeredSpinorFieldType)
    inline data_tp& operator()(const RangesTp auto &x, const int &c) { return store_staggered_spinor_component(Indices{}, x, c); }
             
};



