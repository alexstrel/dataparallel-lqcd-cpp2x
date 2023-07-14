#pragma once
//
#include <core/color_spinor.h>

template<GenericFieldTp F, bool is_constant = false, int bSize = 1>
class FieldAccessor {
  public:
    using data_tp  = typename F::data_tp;

    using SpinorTp = impl::ColorSpinor<data_tp, F::Ncolor(), F::Nspin(), bSize>;
    using LinkTp   = impl::ColorMatrix<data_tp, F::Ncolor(), bSize>;
    
    using Indices  = std::make_index_sequence<F::Ndim()>;      

    using AccessorTp      = typename std::remove_cvref_t< decltype( std::declval<F>().template Accessor<is_constant>()) >; //MDViewTp
    using GhostAccessorTp = typename std::remove_cvref_t< decltype( std::declval<F>().template GhostAccessor<is_constant>( std::declval<const int>())) >; //MDViewTp     
  
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
    template <int nparity = F::Nparity()>
    requires (nparity == 1)
    inline data_tp& operator()(int x, int y, int i) { return field_accessor(x,y,i); }

    /**
       @brief 2-d accessor functor
       @param[in] x coords
       @param[in] y coords
       @param[in] i spin dof or dir       
       @return Complex number at this spin and color index
    */    
    template <int nparity = F::Nparity()>
    requires (nparity == 1)    
    inline const data_tp& operator()(int x, int y, int i) const { return field_accessor(x,y,i); }
    
    /**
       @brief 2-d accessor functor
       @param[in] x coords
       @param[in] y coords
       @param[in] i spin dof or dir      
       @return Complex number at this spin and color index
    */
    template <int nparity = F::Nparity()>
    requires (nparity == 2)
    inline data_tp& operator()(int x, int y, int i, int p) { return field_accessor(x,y,d,p); }

    /**
       @brief 2-d accessor functor
       @param[in] x coords
       @param[in] y coords
       @param[in] i spin dof or dir      
       @return Complex number at this spin and color index
    */
    template <int nparity = F::Nparity()>
    requires (nparity == 2)    
    inline const data_tp& operator()(int x, int y, int i, int p) const { return field_accessor(x,y,d,p); } 
    //
    
    template<std::size_t... Idxs, int nspin = F::Nspin()>
    requires (nspin == 2) 
    inline decltype(auto) load_spinor(std::index_sequence<Idxs...>, const std::array<int, F::Ndim()>& x) const {
    
      std::array<data_tp, F::Ncolor()*F::Nspin()*bSize> spinor{this->field_accessor(x[Idxs]..., 0), this->field_accessor(x[Idxs]..., 1)};
          
      return SpinorTp(spinor);//2-component spinor
    }    
        
    template<std::size_t... Idxs, int ncolor = F::Ncolor()>
    requires (ncolor == 1) 
    inline decltype(auto) load_parity_link(std::index_sequence<Idxs...>, const std::array<int, F::Ndim()>& x, const int &d, const int &parity) const {
    
      std::array<data_tp, F::Ncolor()*F::Ncolor()*bSize> link{this->field_accessor(x[Idxs]..., d, parity)};
      
      return LinkTp(link);
    }      
    
    template<FieldType field_type = F::type()>
    requires (field_type == FieldType::VectorFieldType)    
    inline decltype(auto) operator()(const std::array<int, F::Ndim()> &x, const int &d, const int &p) const {
      return load_parity_link(Indices{}, x, d, p);    
    }            

    template<FieldType field_type = F::type()>
    requires (field_type == FieldType::SpinorFieldType)        
    inline decltype(auto) operator()(const std::array<int, F::Ndim()> &x) const {
      return load_spinor(Indices{}, x);    
    }         
};



