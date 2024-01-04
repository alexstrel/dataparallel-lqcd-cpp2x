#pragma once

#include <memory_resource>

template<GenericStaggeredSpinorFieldTp spinor_tp, typename Arg, bool is_exclusive>
class BlockSpinor; // forward declare to make function definition possible

template <GenericStaggeredSpinorFieldTp spinor_tp, typename Arg, bool is_exclusive = true>
decltype(auto) create_block_spinor(const Arg &arg_, const std::size_t n) {//offset for block spinor

  using container_tp  = spinor_tp::container_tp;
  using data_tp       = container_tp::value_type;

  if constexpr ( is_pmr_allocator_aware_type<container_tp> ) {
    const std::size_t pmr_bytes = (arg_.GetFieldSize())*sizeof(data_tp)*n;

    const bool reserved = true;
    
    auto pmr_buffer = pmr_pool::pmr_malloc<is_exclusive>(pmr_bytes, reserved);
    //
    auto pmr_arg = Arg{arg_, pmr_buffer};
    //
    return BlockSpinor<spinor_tp, Arg, is_exclusive>(pmr_arg, n, reserved);
  } else {
    auto arg = Arg{arg_};
  
    return BlockSpinor<spinor_tp, Arg>(arg, n);
  }
}

template<GenericStaggeredSpinorFieldTp spinor_t, typename SpinorArg, bool is_exclusive = true>
class BlockSpinor{
  public:	
    using block_container_tp = std::vector<spinor_t> ;

    using container_tp  = typename spinor_t::container_tp;
    using arg_tp        = SpinorArg;
    using spinor_tp     = spinor_t;
    using spinor_view_t = decltype(std::declval<spinor_t>().View());    

    block_container_tp v;

    SpinorArg args;

    template<StaggeredSpinorFieldTp T = spinor_t>
    BlockSpinor(const SpinorArg &args, const std::size_t n) : args(args) {
      v.reserve(n);
     
      for(int i = 0; i < n; i++) {
	v.push_back(create_field<container_tp, SpinorArg>(args));      
      }
    }
    
    template<PMRStaggeredSpinorFieldTp T = spinor_t>
    BlockSpinor(const SpinorArg &args_, const std::size_t n, const bool is_reserved) : args(args_) {
      using data_tp = container_tp::value_type;;

      v.reserve(n);

      for(int i = 0; i < n; i++) {
        v.push_back(create_field_with_buffer<container_tp, SpinorArg, is_exclusive>(args, is_reserved));
      }
      //
      args.UpdatedReservedPMR();//now locked
    }

    template <StaggeredSpinorFieldViewTp T = spinor_t>    
    BlockSpinor(const SpinorArg &args_, const std::size_t n) : args(args_) { v.reserve(n); }    

    decltype(auto) ConvertToView() {
      auto block_spinor_view = BlockSpinor<spinor_view_t, decltype(args), is_exclusive>{args, nComponents()};

      auto&& src_v = block_spinor_view.Get();

      for(auto &v_el : v) { src_v.push_back(v_el.View()); }

      return block_spinor_view;     
    }
    
    decltype(auto) ConvertToView() const {
      auto block_spinor_view = BlockSpinor<spinor_view_t, decltype(args), is_exclusive>{args,  nComponents()};

      auto&& src_v = block_spinor_view.Get();

      for(auto &v_el : v) { src_v.push_back(v_el.View()); }

      return block_spinor_view;     
    } 

    decltype(auto) ConvertToParityView(const FieldParity parity ) {
      static_assert(!is_memory_non_owning_type<container_tp>, "Cannot reference a non-owner field!");
      
      if constexpr (spinor_tp::Nparity() != 2) {
        std::cerr << "Cannot get a parity component from a non-full field, exiting...\n" << std::endl;
        std::quick_exit( EXIT_FAILURE );
      }
      
      using spinor_parity_view_t = decltype(std::declval<spinor_t>().ParityView(parity));

      const std::size_t n = nComponents();

      auto block_spinor_parity_view = BlockSpinor<spinor_parity_view_t, decltype(args), is_exclusive>{args, n};

      auto&& src_v = block_spinor_parity_view.Get();

      for(auto &v_el : v) { src_v.push_back(v_el.ParityView(parity)); }

      return block_spinor_parity_view;
    }

    auto ConvertToEvenView() { return ConvertToParityView(FieldParity::EvenFieldParity );}
    auto ConvertToOddView()  { return ConvertToParityView(FieldParity::OddFieldParity  );}
    
    auto EODecompose() {
      static_assert(spinor_tp::Nparity() == 2);

      return std::make_tuple(this->ConvertToEvenView(), this->ConvertToOddView());
    }    

    auto& Get()  { return v; }

    decltype(auto) BlockView() { return std::span{v}; }

    auto GetDims()       const { return args.GetLatticeDims(); }
    auto GetCBDims()     const { return args.GetParityLatticeDims(); }    

    auto GetFieldOrder()  const { return args.order; }    

    auto GetFieldSubset() const { return (spinor_tp::Nparity() == 2 ? FieldSiteSubset::FullSiteSubset : (spinor_tp::Nparity() == 1 ? FieldSiteSubset::ParitySiteSubset : FieldSiteSubset::InvalidSiteSubset)); }

    auto nComponents()   const { return v.size(); } 

    void destroy() {
      
      for(auto &spinor : v) spinor.destroy();
      
      args.ReleasePMRBuffer();
    }

    decltype(auto) ExportArg() { return args; }
    
    decltype(auto) ExportParityArg(const FieldParity parity) { 
      if constexpr (std::remove_cvref_t<SpinorArg>::nparity == 1) {
        std::quick_exit( EXIT_FAILURE );
      }
      //
      constexpr int nDim   = std::remove_cvref_t<SpinorArg>::ndim;      
      constexpr int nDir   = std::remove_cvref_t<SpinorArg>::ndir;            
      constexpr int nColor = std::remove_cvref_t<SpinorArg>::ncolor;
      constexpr int nSpin  = std::remove_cvref_t<SpinorArg>::nspin;      
      
      constexpr std::size_t nParity = 1;
      
      return FieldDescriptor<nDim, nDir, nSpin, nColor, nParity>(this->args, parity);
    }  

    spinor_t& operator[](const std::size_t i) { return v[i]; }
};



