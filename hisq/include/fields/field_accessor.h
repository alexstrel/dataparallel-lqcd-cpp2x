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
    using GhostAccessorTp = typename std::remove_cvref_t< decltype( std::declval<F>().template GhostAccessor<is_constant>( std::declval<const int>())) >;
  
    AccessorTp       field_accessor;
    //
    std::array<GhostAccessorTp, F::Ndim()>  ghost_accessor;    
    
    FieldAccessor(const F &field ) : field_accessor(field.template Accessor<is_constant>()),
                                     ghost_accessor({field.template GhostAccessor<is_constant>(0), 
                                                     field.template GhostAccessor<is_constant>(1)}) {}        

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
    requires (type == StaggeredSpinorFieldType)
    inline decltype(auto) load_parity_spinor(std::index_sequence<Idxs...>, const RangesTp auto& x) const {  
      constexpr int ncolor = F::Ncolor();

      auto spinor = [=]() {
        std::array<data_tp, ncolor*bSize> tmp;
#pragma unroll
        for (int i = 0; i < ncolor; i++) {
#pragma unroll
          for (int b = 0; b < bSize; b++ ) {//bSize = 1
            tmp[i*bSize+b] = this->field_accessor(x[Idxs]..., i);
          }
        } return tmp; } ();    

      return SpinorTp(spinor);
    }
    
    template<FieldType field_type = F::type()>
    requires (field_type == FieldType::SpinorFieldType)        
    inline decltype(auto) operator()(const RangesTp auto &x) const {
      return load_parity_spinor(Indices{}, x);    
    }

    template<std::size_t... Idxs, FieldType = F::Type()>
    requires (type == VectorFieldType)
    inline decltype(auto) load_parity_link(std::index_sequence<Idxs...>, const RangesTp auto& x, const int &d, const int &parity) const {

      constexpr int ncolor = F::Ncolor();

      auto link = [=]() {
        std::array<data_tp, ncolor*ncolor*bSize> tmp;
#pragma unroll
        for (int j = 0; j < ncolor; j++) {
#pragma unroll
          for (int i = 0; i < ncolor; i++) {
#pragma unroll
            for (int b = 0; b < bSize; b++ ) {//bSize = 1
              tmp[(j*ncolor+i)*bSize+b] = this->field_accessor(x[Idxs]..., j, i, d, parity);
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



