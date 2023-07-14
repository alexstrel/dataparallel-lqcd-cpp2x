#pragma once
#include <algorithm>
#include <execution>
//
#include <core/cartesian_product.hpp>
//
#include <fields/field_accessor.h>

template <typename Arg>
class FieldPack{
  public:
    using ArgTp  = typename std::remove_cvref_t<Arg>;

    const Arg &args;

    FieldPack(const Arg &args) : args(args) {}     

};

//DslashTransform
template<typename KernelArgs, template <typename Args> class PackKernel>
class Pack{
  private:
    std::unique_ptr<PackKernel<KernelArgs>> pack_kernel_ptr;
     
  public:
    using kernel_data_tp = typename std::remove_cvref_t<KernelArgs>::gauge_data_tp;
    
    static constexpr std::size_t bSize  = std::remove_cvref_t<KernelArgs>::bSize;

    DslashTransform(const KernelArgs &args) : dslash_kernel_ptr(new Kernel<KernelArgs>(args)) {}
    
    KernelArgs& ExportKernelArgs() const { return dslash_kernel_ptr->args; }
    
    template<bool dagger>
    inline void launch_pack(GenericSpinorFieldViewTp auto &out_view, const GenericSpinorFieldViewTp auto &in_view, const GenericSpinorFieldViewTp auto &aux_view, auto&& post_transformer, const FieldParity parity, const auto ids) {
      
      auto DslashKernel = [=, &dslash_kernel   = *dslash_kernel_ptr] (const auto coords) { 
                            //
                            dslash_kernel.template apply<dagger>(out_view, in_view, aux_view, post_transformer, coords, parity); 
                          };
      //
      std::for_each(std::execution::par_unseq,
                    ids.begin(),
                    ids.end(),
                    DslashKernel);    
    } 
    
    template<bool dagger>
    inline void launch_dslash(GenericSpinorFieldViewTp auto &out_view, const GenericSpinorFieldViewTp auto &in_view, const FieldParity parity, const auto ids) {
      
      auto DslashKernel = [=, &dslash_kernel   = *dslash_kernel_ptr] (const auto coords) { 
                            //
                            dslash_kernel.template apply<dagger>(out_view, in_view, coords, parity); 
                          };
      //
      std::for_each(std::execution::par_unseq,
                    ids.begin(),
                    ids.end(),
                    DslashKernel);    
    }    
 
    void operator()(GenericSpinorFieldTp auto &out, const GenericSpinorFieldTp auto &in, const GenericSpinorFieldTp auto &aux, auto&& post_transformer, const FieldParity parity, const bool dagger = false){
      
      if ( in.GetFieldOrder() != FieldOrder::EOFieldOrder and in.GetFieldSubset() != FieldSiteSubset::ParitySiteSubset ) { 
        std::cerr << "Only parity field is allowed." << std::endl; 
        std::quick_exit( EXIT_FAILURE );  
      }    

      using spinor_tp    = typename std::remove_cvref_t<decltype(in)>;
      using container_tp = spinor_tp::container_tp;
      
      //Setup exe domain
      const auto [Nx, Ny] = out.GetCBDims(); //Get CB dimensions
      
      auto X = std::views::iota(0, Nx);
      auto Y = std::views::iota(0, Ny);

      auto ids = std::views::cartesian_product(Y, X);//Y is the slowest index, X is the fastest
      
      if constexpr (is_allocator_aware_type<container_tp> or is_pmr_allocator_aware_type<container_tp>) {
        auto&& out_view       = out.View();
        const auto&& in_view  = in.View();
        const auto&& aux_view = aux.View();         
        
        if (dagger) {
          launch_dslash<true>(out_view, in_view, aux_view, post_transformer, parity, ids);
        } else {
          launch_dslash<false>(out_view, in_view, aux_view, post_transformer, parity, ids);
        }      
      } else {
        if (dagger) {
          launch_dslash<true>(out, in, aux, post_transformer, parity, ids);  
        } else {
          launch_dslash<false>(out, in, aux, post_transformer, parity, ids);
        }
      }       
    }
    
    void operator()(GenericBlockSpinorFieldTp auto &out_block_spinor, GenericBlockSpinorFieldTp auto &in_block_spinor, GenericBlockSpinorFieldTp auto &aux_block_spinor, auto&& post_transformer, const FieldParity parity, const bool dagger = false){ 
      //   
      assert(in_block_spinor.GetFieldOrder() == FieldOrder::EOFieldOrder and in_block_spinor.GetFieldSubset() == FieldSiteSubset::ParitySiteSubset);
      
      using block_spinor_tp        = typename std::remove_cvref_t<decltype(in_block_spinor)>;
      using component_container_tp = block_spinor_tp::container_tp;      
      
      // Take into account only internal points:
      const auto [Nx, Ny] = in_block_spinor.GetCBDims(); //Get CB dimensions

      auto X = std::views::iota(0, Nx);
      auto Y = std::views::iota(0, Ny);

      auto ids = std::views::cartesian_product(Y, X);//Y is the slowest index, X is the fastest
                    
     if constexpr (is_allocator_aware_type<component_container_tp> or is_pmr_allocator_aware_type<component_container_tp>) {
        //First, we need to convert to views all components in the block
        auto &&out_block_spinor_view    = out_block_spinor.ConvertToView();
        auto &&in_block_spinor_view     = in_block_spinor.ConvertToView();       
        auto &&aux_block_spinor_view  = aux_block_spinor.ConvertToView();               

        auto &&out_view    = out_block_spinor_view.BlockView();
        auto &&in_view     = in_block_spinor_view.BlockView(); 
        auto &&aux_view  = aux_block_spinor_view.BlockView();         
        
        if (dagger) {
          launch_dslash<true>(out_view, in_view, aux_view, post_transformer, parity, ids);  
        } else {
          launch_dslash<false>(out_view, in_view, aux_view, post_transformer, parity, ids);
        }  
      } else {
        auto &&out_view    = out_block_spinor.BlockView();
        auto &&in_view     = in_block_spinor.BlockView(); 
        auto &&aux_view  = aux_block_spinor.BlockView();         
      
        if (dagger) {
          launch_dslash<true>(out_view, in_view, aux_view, post_transformer, parity, ids);  
        } else {
          launch_dslash<false>(out_view, in_view, aux_view, post_transformer, parity, ids);
        }     
      }                    
    }    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
};





